"""Generate or densify a stochastic template bank (v3).

Pipeline per tau0 window: draw proposals (uniform, then KDE rounds) ->
screen every proposal in parallel workers against a fixed bank snapshot ->
optimistically collect survivors -> end-of-round batch dedup among the
survivors -> insert the kept templates -> repeat until the acceptance rate
drops below tolerance -> crawl to the next tau0 window.

v3 architectural change relative to v2 (see git-repo/TODO.md, 2026-07-22):
v2 cross-checked each surviving proposal serially in the parent against the
templates accepted earlier in the same round.  That serial tail grows as
O(survivors x intra-round acceptances) and dominates exactly in
high-acceptance rounds.  v3 instead accepts all survivors optimistically and
runs one batch dedup at the end of the round: pairwise matches among
survivors (tau0/sigma pruned) are computed in parallel, then a greedy
elimination scan in proposal order runs on the cached numbers only.  The
final bank is exactly the one the serial v1 loop would have produced for the
same ordered proposal stream: a proposal is rejected iff it matches above
minimal-match either a snapshot template (worker screen) or an
earlier-kept survivor of the same round (greedy scan).

Other changes relative to v2:
- imap_unordered: no head-of-line blocking behind slow (high-eccentricity)
  waveform generations; proposal order is restored afterwards via indices.
- Atomic checkpoint/state writes (tmp file + os.rename), checkpoint first,
  state second, so a kill can never leave a new state pointing at a
  half-written checkpoint.
- Exact-duplicate filtering when loading multiple input banks: a v2 resume
  loaded the current window's seed templates from both the seed bank and the
  checkpoint, duplicating them in the output (absorbed downstream only by
  merge_v6_densified_bank.py's parameter-tuple dedup).
- Single shared pruning helper (v2 hand-duplicated the logic in
  _screen_match_snapshot and TriangleBank.__contains__).
- Per-template 'max_match' convergence diagnostic (adapted from upstream
  gwastro/pycbc PR #5281): the largest match each accepted template had
  against the final bank at acceptance time, NaN for seed-loaded templates.
  Trailing mean/min are logged per round.  This quantifies whether new
  templates fill genuine holes (low max_match) or polish marginal gaps
  (max_match just below minimal-match).
- The trimmed-buffer peak search is now an explicit option (see
  GenUniformWaveform).
- --psd-file is required; v2 silently fell back to a hard-coded O3 PSD path.
- Dropped dead options: --fixed-params/--fixed-values (parsed, never used)
  and --template-duration-threshold (could never work: the generator never
  produces template_duration).
- KDE proposals no longer include derived keys (template_s); v2 latently fed
  template_s to the KDE and back into the waveform generator.

Checkpoint/resume behaviour is otherwise unchanged from v2: SIGTERM/SIGINT
finish the current round, checkpoint, write <output>.state.json and exit 85;
on start, an existing checkpoint_<output> is auto-loaded as an extra input
bank and the crawl restarts at the recorded tau0.  Run from the intended
output directory (checkpoint files are named relative to --output-file).
"""
import argparse
import json
import logging
import multiprocessing
import os
import signal
import sys
import time

import numpy as np
import h5py
from scipy.stats import gaussian_kde
from tqdm import tqdm

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft
import pycbc.conversions
from pycbc.conversions import tau0_from_mass1_mass2

# Linearized triangle-inequality bound used to skip provably-low matches:
# given match(new, old) = m and match(old, other) = m_oo, the code bounds
# match(new, other) <= m_oo - m + TRIANGLE_MARGIN.  With the exact metric
# (sqrt(1 - m) is the distance, 1 - m is not) the linear form with margin
# 1.0 can be violated; the extra 0.10 slack makes the bound conservative in
# practice.  Inherited unchanged from upstream pycbc_brute_bank; templates
# are only skipped, never accepted, based on this bound, and only when the
# bound already proves the match cannot reach SKIP_THRESHOLD (below).
TRIANGLE_MARGIN = 1.10

# Dedup pair lists shorter than this are matched serially in the parent;
# longer lists go to a worker pool (workers inherit survivor waveforms via
# fork, so nothing large is pickled).
PAIR_SERIAL_LIMIT = 512

# Derived per-template bookkeeping keys that must never be treated as
# sampling dimensions or re-fed to the waveform generator.
DERIVED_KEYS = ('template_s', 'f_lower', 'max_match', 'template_duration',
                'tempalte_s')  # 'tempalte_s': typo key present in old files


def skip_threshold(minimal_match):
    # Neighbors whose bounded best match is below 1 - 2*(1 - MM) can be
    # skipped: the triangle bound then proves the true match is below MM, so
    # the accept/reject verdict at MM stays exact (same guarantee stated in
    # bank_faithfulness_scan.py).
    return 1 - (1 - minimal_match) * 2.0


class GenUniformWaveform(object):
    """Frequency-domain template generator and match engine.

    Physical conventions:
    - Templates are generated at delta_f = 1/buffer_length, whitened by
      1/sqrt(PSD) above f_lower, and normalized to unit sigma.  The
      pre-normalization sigma^2 is stored as params['template_s'] and drives
      the optional sigma prefilter.
    - match() is the standard overlap maximized over relative time and
      phase: complex correlate + IFFT, peak of |q|.  By default the peak is
      searched only in the first and last 100 samples of the IFFT output,
      i.e. relative time shifts within about +-100/sample_rate seconds
      (+-49 ms at 2048 Hz).  This trimmed-buffer convention is how every
      bank in this family (v5, v6, faithfulness scans) has been built; pass
      full_buffer_peak=True to search the whole buffer instead (matches the
      upstream gwastro/pycbc default, see PR #5213).
    """

    def __init__(self, buffer_length, sample_rate, f_lower, psd_path,
                 full_buffer_peak=False):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate)
        self.flen = tlen // 2 + 1

        psd = pycbc.psd.read.from_txt(psd_path, self.flen, self.delta_f,
                                      self.f_lower, is_asd_file=False)

        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(np.float32)

        qtilde = pycbc.types.zeros(tlen, np.complex64)
        q = pycbc.types.zeros(tlen, np.complex64)
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)

        if full_buffer_peak:
            self.peak_segments = (q._data,)
        else:
            self.peak_segments = (q._data[-100:], q._data[0:100])

    def generate(self, **kwds):
        try:
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f=self.delta_f,
                                                   f_lower=self.f_lower,
                                                   **kwds)
        except Exception as e:
            logging.info("Waveform generation failed: %s", e)
            return None

        hp.resize(self.flen)
        hp = hp.astype(np.complex64)
        hp[self.kmin:-1] *= self.w
        s = float(1.0 / pycbc.filter.sigmasq(
            hp, low_frequency_cutoff=self.f_lower) ** 0.5)
        hp *= s
        hp.params = kwds
        hp.view = hp[self.kmin:-1]
        hp.params['template_s'] = (1.0 / s) ** 2.0
        return hp

    def match(self, hp, hc):
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(seg).max() for seg in self.peak_segments)
        return m * 4.0 * self.delta_f


class Shrinker(object):
    def __init__(self, data):
        self.indices = data

    def pop(self):
        if len(self.indices) == 0:
            return None
        l = self.indices[-1]
        self.indices = self.indices[:-1]
        return l


class TriangleBank(object):
    """A bank of templates using triangle-inequality match bounds."""

    def __init__(self, args):
        self.waveforms = []
        self.tbins = {}  # tau0 bin -> list of bank indices
        self.tau0 = np.array([])
        self.tau0_threshold = args.tau0_threshold
        self.tau0_cutoff_frequency = args.tau0_cutoff_frequency
        self.sigma_threshold = args.sigma_threshold
        if self.sigma_threshold:
            self.sigma = np.array([])
        self.nprocesses = args.nprocesses
        self.minimal_match = args.minimal_match

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, index):
        return self.waveforms[index]

    def activelen(self):
        return sum(1 for w in self.waveforms
                   if isinstance(w, pycbc.types.FrequencySeries))

    def keys(self):
        return list(self.waveforms[0].params)

    def key(self, k):
        return np.array([p.params[k] for p in self.waveforms])

    def culltau0(self, threshold):
        """Replace waveforms below the crawl window with lightweight stubs
        (frees the frequency-series data; params kept for the final save)."""
        class Stub(object):
            pass

        t0 = np.array([h.tau0 for h in self])
        for c in np.where(t0 < threshold)[0]:
            d = Stub()
            d.tau0 = self.waveforms[c].tau0
            d.params = self.waveforms[c].params
            d.max_match = getattr(self.waveforms[c], 'max_match', np.nan)
            self.waveforms[c] = d

    def candidate_indices(self, hp):
        """Shared tau0-bin + sigma prefilter (v2 duplicated this logic in
        the screen worker and __contains__).  Sets hp.tau0 / hp.tbin."""
        hp.tau0 = tau0_from_mass1_mass2(hp.params['mass1'],
                                        hp.params['mass2'],
                                        self.tau0_cutoff_frequency)
        hp.tbin = int(hp.tau0 / self.tau0_threshold)
        if hp.tbin in self.tbins:
            match_range = np.array(self.tbins[hp.tbin], dtype=int)
            sel = np.where(abs(self.tau0[match_range] - hp.tau0)
                           < self.tau0_threshold)[0]
            match_range = match_range[sel]
        else:
            match_range = np.array([], dtype=int)
        if self.sigma_threshold:
            sr = self.sigma[match_range] / hp.params['template_s']
            isr = hp.params['template_s'] / self.sigma[match_range]
            sel = np.where(np.maximum(sr, isr) < self.sigma_threshold)[0]
            match_range = match_range[sel]
        return match_range

    def screen(self, hp, engine):
        """Deep check of hp against this bank (used on a fixed snapshot in
        workers).  Returns (covered, matches, indices, mmax) where matches/
        indices record every match actually computed (bookkeeping for later
        triangle bounds) and mmax is the largest computed match, a lower
        bound on the true best match when the verdict is not-covered."""
        neighbor = Shrinker(self.candidate_indices(hp))
        bound = np.ones(len(self))
        skip = skip_threshold(self.minimal_match)
        matches = []
        indices = []
        mmax = 0.0
        while True:
            j = neighbor.pop()
            if j is None:
                return (False, np.array(matches),
                        np.array(indices, dtype=int), mmax)
            oldhp = self[j]
            m = engine.match(hp, oldhp)
            if m > self.minimal_match:
                return True, None, None, m
            if m > mmax:
                mmax = m
            indices.append(j)
            matches.append(m)
            bound[j] = m
            est = oldhp.maxmatch_matrix_r - m + TRIANGLE_MARGIN
            upd = np.where(est < bound[oldhp.indices])[0]
            bound[oldhp.indices[upd]] = est[upd]
            neighbor.indices = neighbor.indices[
                bound[neighbor.indices] > skip]

    def insert(self, hp):
        self.waveforms.append(hp)
        # Register under the neighboring bins too so candidate_indices only
        # needs to look up a proposal's own bin.
        for b in [hp.tbin - 1, hp.tbin, hp.tbin + 1]:
            self.tbins.setdefault(b, []).append(len(self) - 1)
        self.tau0 = np.append(self.tau0, hp.tau0)
        if self.sigma_threshold:
            self.sigma = np.append(self.sigma, hp.params['template_s'])

    # -- proposal round ----------------------------------------------------

    def check_params(self, params):
        """Screen one round of proposals in parallel, then batch-dedup the
        survivors.  Result is identical to the v1 serial loop for the same
        ordered proposal stream (see module docstring)."""
        total_num = len(tuple(params.values())[0])
        n_covered = 0
        n_failed = 0
        global _screen_bank
        _screen_bank = self  # workers inherit via fork, never pickled
        survivors = []
        with multiprocessing.Pool(
                self.nprocesses, initializer=_screen_init,
                initargs=(gen.init_args,)) as pool:
            results = pool.imap_unordered(
                _screen_worker,
                ((idx, {k: params[k][idx] for k in params})
                 for idx in range(total_num)),
                chunksize=8)
            for idx, res in results:
                if res is None:
                    n_failed += 1
                elif res == 'covered':
                    n_covered += 1
                else:
                    survivors.append((idx, res))
        _screen_bank = None
        survivors.sort(key=lambda t: t[0])  # restore proposal order
        num_added, n_elim = self.dedup_and_insert(survivors, total_num)
        logging.info("round summary: %i proposed, %i covered in workers, "
                     "%i gen-failed, %i survivors, %i eliminated in dedup, "
                     "%i added", total_num, n_covered, n_failed,
                     len(survivors), n_elim, num_added)
        return self, num_added / total_num

    def dedup_and_insert(self, survivors, total_num):
        """End-of-round batch dedup: parallel pairwise matches among the
        survivors, then a greedy elimination scan in proposal order on the
        cached values.  Keeps exactly the survivors the serial loop would
        have accepted."""
        n = len(survivors)
        if n == 0:
            return 0, 0
        hps = [hp for _, hp in survivors]

        # Enumerate candidate pairs with the same tau0/sigma criteria the
        # bank itself uses (pairs outside them are below minimal match by
        # construction).  Two-pointer sweep over tau0-sorted order keeps
        # this O(n log n + pairs).
        tau0 = np.array([hp.tau0 for hp in hps])
        order = np.argsort(tau0, kind='stable')
        pairs = []
        for oi, a in enumerate(order):
            for b in order[oi + 1:]:
                if tau0[b] - tau0[a] >= self.tau0_threshold:
                    break
                if self.sigma_threshold:
                    sa = hps[a].params['template_s']
                    sb = hps[b].params['template_s']
                    if max(sa / sb, sb / sa) >= self.sigma_threshold:
                        continue
                pairs.append((min(a, b), max(a, b)))

        pair_match = self._pairwise_matches(hps, pairs)

        # Greedy elimination in proposal order: survivor b dies iff an
        # earlier-kept survivor a covers it -- exactly the serial
        # cross-check, but running on cached numbers only.
        kept = []
        intra = {}
        for b in range(n):
            covering = None
            recorded = []
            for a in kept:
                m = pair_match.get((min(a, b), max(a, b)))
                if m is None:
                    continue
                if m > self.minimal_match:
                    covering = a
                    break
                recorded.append((a, m))
            if covering is None:
                intra[b] = recorded
                kept.append(b)

        # Insert in proposal order, folding the dedup matches into the
        # triangle bookkeeping and the max_match diagnostic.
        bank_index = {}
        for b in kept:
            hp = hps[b]
            extra = intra[b]
            if extra:
                hp.indices = np.concatenate(
                    [hp.indices,
                     np.array([bank_index[a] for a, _ in extra], dtype=int)])
                hp.maxmatch_matrix_r = np.concatenate(
                    [hp.maxmatch_matrix_r,
                     np.array([m for _, m in extra])])
            hp.max_match = max([hp.screen_mmax] + [m for _, m in extra])
            bank_index[b] = len(self)
            self.insert(hp)
            logging.info("Add (%i/%i) into the bank. BankSize:%i "
                         "MaxMatch:%0.3f", hp.proposal_idx + 1, total_num,
                         len(self), hp.max_match)
        return len(kept), n - len(kept)

    def _pairwise_matches(self, hps, pairs):
        if not pairs:
            return {}
        if len(pairs) <= PAIR_SERIAL_LIMIT:
            return {(a, b): gen.match(hps[a], hps[b]) for a, b in pairs}
        global _dedup_hps
        _dedup_hps = hps  # inherited by workers via fork
        nchunks = self.nprocesses * 4
        chunks = [pairs[i::nchunks] for i in range(nchunks)]
        chunks = [c for c in chunks if c]
        out = {}
        with multiprocessing.Pool(
                self.nprocesses, initializer=_screen_init,
                initargs=(gen.init_args,)) as pool:
            for part in pool.imap_unordered(_dedup_worker, chunks):
                out.update(part)
        _dedup_hps = None
        return out

    # -- input banks -------------------------------------------------------

    def add_existing_bank(self, params, seen):
        """Force-add templates from an input bank.  `seen` is a set of
        parameter tuples already loaded this window; exact duplicates are
        skipped (a v2 resume loaded the current window's seed templates from
        both the seed bank and the checkpoint)."""
        total_num = len(tuple(params.values())[0])
        param_keys = sorted(k for k in params if k != 'approximant')
        fresh = []
        n_dupes = 0
        for idx in range(total_num):
            fp = tuple(params[k][idx] for k in param_keys)
            if fp in seen:
                n_dupes += 1
                continue
            seen.add(fp)
            fresh.append({k: params[k][idx] for k in params})
        if n_dupes:
            logging.info("skipped %i exact-duplicate input templates",
                         n_dupes)

        # Workers inherit the parent generator via fork.  generate() never
        # executes the shared IFFT plan (only match() does), so no
        # per-worker engine is needed here.
        with multiprocessing.Pool(self.nprocesses) as pool:
            waveform_cache = list(pool.imap_unordered(
                wf_wrapper, tqdm(fresh)))

        for i, hp in enumerate(waveform_cache):
            if hp is None:
                logging.info("#%i input waveform generation failed!", i)
                continue
            hp.tau0 = tau0_from_mass1_mass2(hp.params['mass1'],
                                            hp.params['mass2'],
                                            self.tau0_cutoff_frequency)
            hp.tbin = int(hp.tau0 / self.tau0_threshold)
            hp.maxmatch_matrix_r = np.array([])
            hp.indices = np.array([], dtype=int)
            self.insert(hp)
        return self


# -- pool workers (fork-inherited globals) ----------------------------------

_screen_bank = None   # bank snapshot for screen workers
_screen_gen = None    # per-worker match engine (FFT plans are not fork-safe)
_dedup_hps = None     # survivor list for dedup workers


def _screen_init(gen_args):
    global _screen_gen
    _screen_gen = GenUniformWaveform(*gen_args)


def _screen_worker(item):
    idx, p = item
    try:
        hp = _screen_gen.generate(**p)
    except Exception as e:
        logging.info("Waveform generation failed: %s", e)
        hp = None
    if hp is None:
        return idx, None
    covered, matches, indices, mmax = _screen_bank.screen(hp, _screen_gen)
    if covered:
        return idx, 'covered'
    hp.maxmatch_matrix_r = matches
    hp.indices = indices
    hp.screen_mmax = mmax
    hp.proposal_idx = idx
    return idx, hp


def _dedup_worker(pair_chunk):
    return {(a, b): _screen_gen.match(_dedup_hps[a], _dedup_hps[b])
            for a, b in pair_chunk}


def wf_wrapper(p):
    try:
        return gen.generate(**p)
    except Exception as e:
        logging.info("Waveform generation failed: %s", e)
        return None


# -- graceful stop / checkpointing ------------------------------------------

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True
    logging.info("signal %s received: will checkpoint and exit at the next "
                 "safe point", signum)


def _atomic_write_json(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(payload, fh)
    os.rename(tmp, path)


def save_bank(args, bank, checkpoint=False):
    """Write the bank (templates within [tau0_start, tau0_end]) atomically:
    write to a temp file, then rename.  A kill mid-write can therefore never
    corrupt an existing checkpoint."""
    path = ('checkpoint_' + args.output_file) if checkpoint \
        else args.output_file
    if len(bank) == 0:
        if checkpoint:
            logging.info("No waveforms generated. Skipping checkpoint.")
            return
        logging.info("No waveforms generated. Exiting.")
        sys.exit()

    t = tau0_from_mass1_mass2(bank.key('mass1'), bank.key('mass2'),
                              args.tau0_cutoff_frequency)
    l = (t <= args.tau0_end) & (t >= args.tau0_start)
    tmp = path + '.tmp'
    with h5py.File(tmp, 'w') as o:
        o.attrs['minimal_match'] = args.minimal_match
        for k in bank.keys():
            val = bank.key(k)[l]
            if val.dtype.char == 'U':
                val = val.astype('bytes')
            o[k] = val
        o['f_lower'] = np.array([args.low_frequency_cutoff] * int(l.sum()))
        o['max_match'] = np.array(
            [getattr(w, 'max_match', np.nan) for w in bank.waveforms])[l]
    os.rename(tmp, path)


def checkpoint_and_state(args, bank, tau0s, tau0e):
    # Checkpoint first, state second (both atomic): a kill in between leaves
    # the old state pointing at a valid (newer) checkpoint, which only costs
    # re-crawling one region.
    save_bank(args, bank, checkpoint=True)
    _atomic_write_json(args.output_file + '.state.json',
                       {'tau0s': tau0s, 'tau0e': tau0e})


_last_checkpoint = time.time()


def maybe_checkpoint(args, bank, tau0s, tau0e):
    """Time-based checkpoint plus the graceful-stop exit point."""
    global _last_checkpoint
    if _stop_requested:
        logging.info("stop requested: checkpointing and exiting 85")
        checkpoint_and_state(args, bank, tau0s, tau0e)
        sys.exit(85)
    if time.time() - _last_checkpoint > args.checkpoint_time:
        logging.info("Checkpointing bank")
        checkpoint_and_state(args, bank, tau0s, tau0e)
        _last_checkpoint = time.time()


# -- proposal drawing --------------------------------------------------------

def draw(rtype, args, bank):
    """Draw one batch of random proposal parameters."""
    if rtype == 'uniform':
        params = {name: np.random.uniform(pmin, pmax, size=args.size)
                  for name, pmin, pmax
                  in zip(args.params, args.min, args.max)}
    elif rtype == 'kde':
        trail = min(300, len(bank))
        p = [k for k in bank.keys()
             if k != 'approximant' and k not in DERIVED_KEYS]
        bdata = np.array([bank.key(k)[-trail:] for k in p])
        kde = gaussian_kde(bdata)
        points = kde.resample(size=args.size)
        params = {k: v for k, v in zip(p, points)}

    params['approximant'] = np.array([args.approximant] * args.size)

    # Boundary and constraint filters (the KDE can propose outside bounds).
    l = None
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        nl = (params[name] < pmax) & (params[name] > pmin)
        l = (nl & l) if l is not None else nl
    if args.max_q:
        q = np.maximum(params['mass1'] / params['mass2'],
                       params['mass2'] / params['mass1'])
        l &= q < args.max_q

    # The eccentricity and spin caps below interpolate linearly in chirp
    # mass between the knots mchirp(5,5) = 4.352752816480621,
    # mchirp(15,15) = 13.058258449441862 and
    # mchirp(30,30) = 26.116516898883724 (literals kept to preserve exact
    # floating-point behaviour of the produced banks).
    if args.ecc_constraint:
        mchirp15 = 13.058258449441862
        mchirp5 = 4.352752816480621
        mc = pycbc.conversions.mchirp_from_mass1_mass2(params['mass1'],
                                                       params['mass2'])
        max_ecc = np.where(
            mc <= mchirp5, 0.3,
            np.where(mc >= mchirp15, 0.5,
                     0.3 + (0.5 - 0.3) * (mc - mchirp5)
                     / (mchirp15 - mchirp5)))
        l &= params['eccentricity'] < max_ecc

    if args.spin_constraint:
        mchirp15 = 13.058258449441862
        mchirp30 = 26.116516898883724
        mc = pycbc.conversions.mchirp_from_mass1_mass2(params['mass1'],
                                                       params['mass2'])
        max_spin = np.where(
            mc <= mchirp15, 0.5,
            np.where(mc >= mchirp30, 0.8,
                     0.5 + (0.8 - 0.5) * (mc - mchirp15)
                     / (mchirp30 - mchirp15)))
        l &= params['spin1z'] < max_spin
        l &= params['spin2z'] < max_spin

    return {k: params[k][l] for k in params}


def cdraw(rtype, ts, te, args, bank):
    """Draw proposals until args.size of them land in the tau0 window."""
    def in_window(p):
        if len(p[list(p.keys())[0]]) == 0:
            return p
        t = tau0_from_mass1_mass2(p['mass1'], p['mass2'],
                                  args.tau0_cutoff_frequency)
        l = (t < te) & (t > ts)
        return {k: p[k][l] for k in p}

    p = in_window(draw(rtype, args, bank))
    i = 0
    while len(p[list(p.keys())[0]]) < args.size:
        tp = in_window(draw(rtype, args, bank))
        p = {k: np.concatenate([p[k], tp[k]]) for k in p}
        i += 1
        if i > args.placement_iterations:
            break
    if len(p[list(p.keys())[0]]) == 0:
        return None
    return p


def adjustmass(args, tau0s, tau0e):
    """Shrink the sampled mass range to what can reach this tau0 window."""
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        if name == 'mass1':
            mass1min, mass1max = pmin, pmax
        elif name == 'mass2':
            mass2min, mass2max = pmin, pmax

    massmin = min(mass1min, mass2min)
    massmax = max(mass1max, mass2max)

    while tau0_from_mass1_mass2(massmin,
                                min(massmin * args.max_q, massmax),
                                args.tau0_cutoff_frequency) > tau0e \
            and massmin < massmax:
        massmin += 0.1
    while tau0_from_mass1_mass2(massmax,
                                max(massmax / args.max_q, massmin),
                                args.tau0_cutoff_frequency) < tau0s \
            and massmax > massmin:
        massmax -= 0.1

    for i, name in enumerate(args.params):
        if name in ('mass1', 'mass2'):
            args.min[i] = max(massmin - 1, mass1min)
            args.max[i] = min(massmax + 1, mass1max)
    return args.min, args.max


# -- main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pycbc.add_common_pycbc_options(parser)
    pycbc.psd.insert_psd_option_group(parser)

    parser.add_argument('--input-file', nargs='*',
        help='Bank(s) to use as a starting point.')
    parser.add_argument('--output-file', required=True,
        help='Output file name for template bank.')

    # parameter ranges
    parser.add_argument('--params', nargs='+',
        help='list of parameters to use')
    parser.add_argument('--min', nargs='+', type=float,
        help='list of the minimum parameter values')
    parser.add_argument('--max', nargs='+', type=float,
        help='list of the maximum parameter values')
    parser.add_argument('--approximant', required=True, type=str,
        help='The waveform approximant to place')
    parser.add_argument('--max-q', type=float, help='maximum mass ratio')
    parser.add_argument('--ecc-constraint', action='store_true',
        help='eccentricity-vs-mchirp cap (see draw())')
    parser.add_argument('--spin-constraint', action='store_true',
        help='spin-vs-mchirp cap (see draw())')

    # waveform parameters
    parser.add_argument('--buffer-length', default=4, type=float,
        help='waveform buffer in seconds; must exceed the longest waveform')
    parser.add_argument('--sample-rate', default=2048, type=float)
    parser.add_argument('--low-frequency-cutoff', default=20.0, type=float)
    parser.add_argument('--full-buffer-peak', action='store_true',
        help='Search the whole IFFT buffer for the match peak instead of '
             'the first/last 100 samples (upstream default). All banks in '
             'this family use the trimmed convention; do not mix within '
             'one bank.')
    parser.add_argument('--nprocesses', type=int, default=1)

    # generation proposal
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--minimal-match', default=0.97, type=float)
    parser.add_argument('--placement-iterations', default=1000, type=int,
        help='max draw attempts per proposal batch')
    parser.add_argument('--tolerance', type=float, required=True,
        help='stop a window when the acceptance rate drops below this')
    parser.add_argument('--size', type=int, required=True,
        help='number of proposals per round')

    # splitting parameters
    parser.add_argument('--sigma-threshold', type=float)
    parser.add_argument('--tau0-threshold', type=float, required=True,
        help='tau0 separation beyond which a match is assumed impossible')
    parser.add_argument('--tau0-crawl', type=float,
        help='tau0 window step length')
    parser.add_argument('--tau0-start', type=float)
    parser.add_argument('--tau0-end', type=float)
    parser.add_argument('--tau0-cutoff-frequency', type=float, default=15.0)
    parser.add_argument('--adjust-mass', action='store_true',
        help='shrink mass range to each tau0 window')
    parser.add_argument('--crawl-one-tau', action='store_true',
        help='advance a full window (instead of half) per crawl step')

    # checkpointing
    parser.add_argument('--checkpoint-time', type=float, default=5000,
        help='seconds between periodic checkpoints')
    args = parser.parse_args()

    if not args.psd_file:
        parser.error("--psd-file is required (v2's silent fallback to a "
                     "hard-coded O3 PSD has been removed)")

    np.random.seed(args.seed)

    # Graceful stop on SIGTERM/SIGINT (condor_rm / eviction / Ctrl-C):
    # finish the current round, checkpoint, exit 85.
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.handlers.clear()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Auto-resume: an existing checkpoint of this output becomes an extra
    # input bank and the crawl restarts at the recorded tau0.
    ckpt_file = 'checkpoint_' + args.output_file
    state_file = args.output_file + '.state.json'
    resume_tau0 = None
    if os.path.exists(ckpt_file):
        args.input_file = (args.input_file or []) + [ckpt_file]
        logging.info("auto-resume: loading checkpoint %s", ckpt_file)
        if os.path.exists(state_file):
            with open(state_file) as fh:
                resume_tau0 = json.load(fh).get('tau0s')
            logging.info("auto-resume: restarting crawl at tau0s=%s",
                         resume_tau0)

    global gen
    gen = GenUniformWaveform(args.buffer_length, args.sample_rate,
                             args.low_frequency_cutoff, args.psd_file,
                             args.full_buffer_peak)
    gen.init_args = (args.buffer_length, args.sample_rate,
                     args.low_frequency_cutoff, args.psd_file,
                     args.full_buffer_peak)
    bank = TriangleBank(args)

    # Reachable tau0 range given the mass bounds.
    mlim = {}
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        if name in ('mass1', 'mass2'):
            mlim[name] = [pmin, pmax]
    taumax = tau0_from_mass1_mass2(mlim['mass1'][0], mlim['mass2'][0],
                                   args.tau0_cutoff_frequency)
    taumin = tau0_from_mass1_mass2(mlim['mass1'][1], mlim['mass2'][1],
                                   args.tau0_cutoff_frequency)

    seen_input = set()
    tau0s = args.tau0_start if resume_tau0 is None else resume_tau0
    tau0e = tau0s + args.tau0_crawl
    logging.info("Starting crawl: tau0s=%s tau0e=%s tau0end=%s",
                 tau0s, tau0e, args.tau0_end)
    taubanke = None
    while tau0e <= args.tau0_end + 1e-8:
        logging.info("tau0 window: %s-%s", tau0s, tau0e)

        if args.input_file:
            if taubanke is None:
                taubanks = tau0s - args.tau0_threshold
                taubanke = tau0e + args.tau0_threshold
            else:
                taubanks = taubanke
                taubanke = tau0e + args.tau0_threshold
            logging.info("Adding existing bank(s) in tau0 range "
                         "%3.2f-%3.2f", taubanks, taubanke)
            ilength = len(bank)
            for bankf in args.input_file:
                with h5py.File(bankf, 'r') as f:
                    if len(f.keys()) == 0:
                        logging.info("Empty file %s", bankf)
                        continue
                    t = tau0_from_mass1_mass2(f['mass1'][()],
                                              f['mass2'][()],
                                              args.tau0_cutoff_frequency)
                    l = (t <= taubanke) & (t >= taubanks)
                    params = {k: f[k][l] for k in f.keys()
                              if k not in DERIVED_KEYS}
                    params['approximant'] = np.array(
                        [v.decode() for v in params['approximant']])
                if len(tuple(params.values())[0]) > 0:
                    logging.info('Adding %s waveforms from %s',
                                 len(tuple(params.values())[0]), bankf)
                    bank = bank.add_existing_bank(params, seen_input)
            logging.info("Existing banks added, banksize: %s (+%s)",
                         len(bank), len(bank) - ilength)

        if args.adjust_mass:
            args.min, args.max = adjustmass(args, tau0s, tau0e)
        for name, pmin, pmax in zip(args.params, args.min, args.max):
            logging.info("parameter %s: %3.3f-%3.3f", name, pmin, pmax)

        accept = 1
        loop = 0
        while accept > args.tolerance and tau0s < taumax and tau0e > taumin:
            # Uniform round
            loop += 1
            params = cdraw('uniform', tau0s, tau0e, args, bank)
            if params is None:
                break

            blen = len(bank)
            bank, uaccept = bank.check_params(params)
            log_round(bank, tau0s, tau0e, 'uniform', loop, loop,
                      uaccept, len(bank) - blen)
            maybe_checkpoint(args, bank, tau0s, tau0e)

            # Acceptance is only trusted once the window has warmed up.
            if loop > 10:
                accept = uaccept

            # KDE rounds focus proposals near recent acceptances; keep
            # going while they stay at least half as productive as the
            # first one.
            kloop = 0
            kaccept = 1
            initial_kaccept = 1
            while ((kloop == 0) or (kaccept / initial_kaccept) > .5) \
                    and len(bank) > 10:
                loop += 1
                kloop += 1
                params = cdraw('kde', tau0s, tau0e, args, bank)
                blen = len(bank)
                bank, kaccept = bank.check_params(params)
                if kloop == 1:
                    initial_kaccept = kaccept
                log_round(bank, tau0s, tau0e, 'kde', kloop, loop,
                          kaccept, len(bank) - blen)
                if kaccept <= args.tolerance:
                    accept = kaccept
                    break
                maybe_checkpoint(args, bank, tau0s, tau0e)

        bank.culltau0(tau0s - args.tau0_threshold * 2.0)
        logging.info("Region Done %.1f-%.1f, %i stored",
                     tau0s, tau0e, bank.activelen())
        next_tau0s = tau0s + (args.tau0_crawl if args.crawl_one_tau
                              else args.tau0_crawl / 2)
        _atomic_write_json(state_file, {'tau0s': next_tau0s, 'tau0e': tau0e})

        if args.crawl_one_tau:
            tau0s += args.tau0_crawl
            tau0e += args.tau0_crawl
        else:
            tau0s += args.tau0_crawl / 2
            tau0e += args.tau0_crawl / 2

    save_bank(args, bank)


def log_round(bank, tau0s, tau0e, kind, kloop, loop, accept, added):
    """Round log line with the trailing max_match diagnostic (upstream
    #5281): mean/min of the last 10% of accepted templates' max_match.
    Trailing values hugging minimal-match mean the round is polishing
    marginal gaps rather than filling holes."""
    mm = np.array([getattr(w, 'max_match', np.nan)
                   for w in bank.waveforms])
    mm = mm[~np.isnan(mm)]
    if len(mm):
        trail = mm[int(len(mm) * 0.9):]
        diag = "trail_maxmatch mean:%0.4f min:%0.4f" % (trail.mean(),
                                                        trail.min())
    else:
        diag = "trail_maxmatch: n/a"
    logging.info("tau0 %3.1f-%3.1f: %s(round %i, total %i) banksize: %i "
                 "accept: %s added: %i %s\n", tau0s, tau0e, kind, kloop,
                 loop, len(bank), accept, added, diag)


if __name__ == '__main__':
    main()
