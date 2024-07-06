#!/usr/bin/env python

# Copyright (C) 2017 Alex Nitz, Duncan Macleod
#               2022 Shichao Wu
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Generate a bank of templates using a brute force stochastic method.
"""
import numpy
import h5py
import logging
import argparse
import pickle
import numpy.random
from scipy.stats import gaussian_kde

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft, pycbc.conversions
import pycbc.pool
from pycbc import transforms
from pycbc.waveform.spa_tmplt import spa_length_in_time
from pycbc.distributions import read_params_from_config
from pycbc.distributions.utils import draw_samples_from_config, prior_from_config
import lal

parser = argparse.ArgumentParser(description=__doc__)
pycbc.add_common_pycbc_options(parser)
parser.add_argument('--output-file', required=True,
    help='Output file name for template bank.')
parser.add_argument('--input-file',
    help='Bank to use as a starting point.')
parser.add_argument('--input-config',
    help='Draw parameters from the given configure file.')
parser.add_argument('--params',
    help='list of paramaters to use', nargs='+')
parser.add_argument('--min',
    help='list of the minimum parameter values', nargs='+', type=float)
parser.add_argument('--max',
    help='list of the maximum parameter values', nargs='+', type=float)
parser.add_argument('--approximant',  required=True,
    help='The waveform approximant to place')
parser.add_argument('--minimal-match', default=0.97, type=float, 
    help='minimal match of SNR due to discreteness of the template bank')
parser.add_argument('--buffer-length', default=2, type=float,
    help='size of waveform buffer in seconds')
parser.add_argument('--max-signal-length', type= float, 
    help="When specified, it cuts the maximum length of the waveform model to the lengh provided")
parser.add_argument('--sample-rate', default=2048, type=float,
    help='sample rate in seconds')
parser.add_argument('--low-frequency-cutoff', default=20.0, type=float)
parser.add_argument('--enable-sigma-bound', action='store_true')
parser.add_argument('--tau0-threshold', type=float, help='threshold to separate two waveforms')
parser.add_argument('--permissive', action='store_true',
    help='Allow waveform generator to fail.')
parser.add_argument('--placement-iterations', default=1000, type=int, 
    help='Specify the number of attempts the bank should make when placing points. Use this option if the bank fails to place any points.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--tolerance', type=float)
parser.add_argument('--max-mtotal', type=float)
parser.add_argument('--min-mchirp', type=float, help='minimum chirp mass')
parser.add_argument('--max-mchirp', type=float, help='maximum chirp mass')
parser.add_argument('--fixed-params', type=str, nargs='*')
parser.add_argument('--fixed-values', type=float, nargs='*')
parser.add_argument('--max-q', type=float, help='maximum mass ratio')
parser.add_argument('--tau0-crawl', type=float, help='step length tau0 would proceed')
parser.add_argument('--tau0-start', type=float, help='starting value for tau0')
parser.add_argument('--tau0-end', type=float, help='ending value for tau0')
parser.add_argument('--tau0-cutoff-frequency', type=float, default=15.0)
parser.add_argument('--nprocesses', type=int, default=1,
    help='Number of processes to use for waveform generation parallelization. If not given then only a single core will be used.')
pycbc.psd.insert_psd_option_group(parser)
args = parser.parse_args()

logger = logging.getLogger()
logger.handlers.clear() # Clear existing handlers
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

#pycbc.init_logging(args.verbose)
numpy.random.seed(args.seed)

# Read the .ini file if it's in the input.
if args.input_config is not None:
    config_parser = pycbc.types.config.InterpolatingConfigParser()
    file = open(args.input_config, 'r')
    config_parser.read_file(file)
    file.close()

    variable_args, static_args = read_params_from_config(
        config_parser, prior_section='prior',
        vargs_section='variable_params',
        sargs_section='static_params')

    if any(config_parser.get_subsections('waveform_transforms')):
        waveform_transforms = transforms.read_transforms_from_config(
                config_parser, 'waveform_transforms')
    else:
        waveform_transforms = None

    dists_joint = prior_from_config(cp=config_parser)

fdict = {}
if args.fixed_params:
    fdict = {p: v for (p, v) in zip(args.fixed_params, args.fixed_values)}

class Shrinker(object):
    def __init__(self, data):
        self.data = data

    def pop(self):
        if len(self.data) == 0:
            return None
        l = self.data[-1]
        self.data = self.data[:-1]
        return l

class TriangleBank(object):
    """ A bank of templates that uses the triangle inequality to estimate
    matches based on prior ones.
    """
    def __init__(self, p=None):
        self.waveforms = p if p is not None else []
        self.tbins = {}

    def __len__(self):
        return len(self.waveforms)

    def activelen(self):
        i = 0
        for w in self.waveforms:
            if isinstance(w, pycbc.types.FrequencySeries):
                i += 1
        return i

    def insert(self, hp):
        self.waveforms.append(hp)

        for b in [hp.tbin - 1, hp.tbin, hp.tbin + 1]:
            if b in self.tbins:
                self.tbins[b].append(len(self)-1)
            else:
                self.tbins[b] = [len(self)-1]

    def __getitem__(self, index):
        return self.waveforms[index]

    def keys(self):
        return self.waveforms[0].params.keys()

    def key(self, k):
        return numpy.array([p.params[k] for p in self.waveforms])

    def sigma_match_bound(self, sig):
        if not hasattr(self, 'sigma'):
            self.sigma = None
        if self.sigma is None or len(self.sigma) != len(self):
            self.sigma = numpy.array([h.s for h in self.waveforms])
        return numpy.minimum(sig / self.sigma, self.sigma / sig)

    def range(self):
        if not hasattr(self, 'r'):
            self.r = None
        if self.r is None or len(self.r) != len(self):
            self.r = numpy.arange(0, len(self))
        return self.r

    def culltau0(self, threshold):
        cull = numpy.where(self.tau0() < threshold)[0]

        class dumb(object):
            pass
        for c in cull:
            d = dumb()
            d.tau0 = self.waveforms[c].tau0
            d.params = self.waveforms[c].params
            d.s = self.waveforms[c].s
            self.waveforms[c] = d

    def tau0(self):
        if not hasattr(self, 't0'):
            self.t0 = None
        if self.t0 is None or len(self.t0) != len(self):
            self.t0 = numpy.array([h.tau0 for h in self])
        return self.t0

    def __contains__(self, newhp):
        # newhp is the newly added waveform
        mmax = 0
        mnum = 0
        #Apply sigmas maximal match.
        if args.enable_sigma_bound:
            matches = self.sigma_match_bound(newhp.s)
            r = self.range()[matches > newhp.threshold]
        else:
            matches = numpy.ones(len(self))
            r = self.range()

        msig = len(r)

        #Apply tau0 threshold
        if args.tau0_threshold:
            newhp.tau0 = pycbc.conversions.tau0_from_mass1_mass2(
                                            newhp.params['mass1'],
                                            newhp.params['mass2'],
                                            args.tau0_cutoff_frequency)
            newhp.tbin = int(newhp.tau0 / args.tau0_threshold)

            if newhp.tbin in self.tbins:
                r = numpy.array(self.tbins[newhp.tbin])
            else:
                r = r[:0]

        mtau = len(r)

        # Try to do some actual matches
        inc = Shrinker(r*1)
        while 1:
            j = inc.pop()
            if j is None:
                newhp.matches = matches[r]
                newhp.indices = r
                logging.info("Add (%i/%i) into the bank. BankSize:%i "
                             "AfterSigma:%i AfterTau0:%i AfterTriangle:%i, MaxMatch:%0.3f"
                              % (newhp.num_tried, newhp.total_num,
                                 len(self), msig, mtau, mnum, mmax))
                return False

            oldhp = self[j]
            m = newhp.gen.match(newhp, oldhp)
            matches[j] = m
            mnum += 1

            # Update bounding match values, apply triangle inequality
            maxmatches = oldhp.matches - m + 1.10
            update = numpy.where(maxmatches < matches[oldhp.indices])[0]
            matches[oldhp.indices[update]] = maxmatches[update]

            # Update where to calculate matches
            skip_threshold = 1 - (1 - newhp.threshold) * 2.0
            inc.data = inc.data[matches[inc.data] > skip_threshold]

            if m > newhp.threshold:
                return True
            if m > mmax:
                mmax = m
    
    def check_params(self, gen, params, threshold):
        num_added = 0
        total_num = len(tuple(params.values())[0])
        waveform_cache = []

        pool = pycbc.pool.choose_pool(args.nprocesses)
        for return_wf in pool.imap_unordered(
                wf_wrapper,
                ({k: params[k][idx] for k in params} for idx in range(total_num))
            ):
            waveform_cache += [return_wf]
        del pool

        for i, hp in enumerate(waveform_cache):
            if hp is not None:
                hp.gen = gen
                hp.threshold = threshold
                hp.total_num = total_num
                hp.num_tried = i + 1
                if hp not in self:
                    num_added += 1
                    self.insert(hp)
            else:
                logging.info("Waveform generation failed!")
                continue

        return bank, num_added / total_num

class GenUniformWaveform(object):
    def __init__(self, buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate)
        self.flen = tlen // 2 + 1
        psd = pycbc.psd.from_cli(args, self.flen, self.delta_f, self.f_lower)
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(numpy.float32)
        qtilde = pycbc.types.zeros(tlen, numpy.complex64)
        q = pycbc.types.zeros(tlen, numpy.complex64)
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        self.md = q._data[-100:]
        self.md2 = q._data[0:100]

    def generate(self, **kwds):  
        kwds.update(fdict)
        if args.max_signal_length is not None:
                flow = numpy.arange(self.f_lower, 100, .1)[::-1]
                length = spa_length_in_time(mass1=kwds['mass1'], mass2=kwds['mass2'], f_lower=flow, phase_order=-1)
                maxlen = args.max_signal_length
                x = numpy.searchsorted(length, maxlen) - 1
                l = length[x]
                f = flow[x]
        else:
                f = self.f_lower
        
        kwds['f_lower'] = f
        
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, hc = pycbc.waveform.get_fd_waveform(delta_f=self.delta_f,
                                                    **kwds)
            
            
            if 'fratio' in kwds:
                hp = hc * kwds['fratio'] + hp * (1 - kwds['fratio'])
                    
        else:
            dt = 1.0 / args.sample_rate
            hp = pycbc.waveform.get_waveform_filter(
                        pycbc.types.zeros(self.flen, dtype=numpy.complex64),
                        delta_f=self.delta_f, delta_t=dt,
                        **kwds)

        hp.resize(self.flen)
        hp = hp.astype(numpy.complex64)
        hp[self.kmin:-1] *= self.w
        s = float(1.0 / pycbc.filter.sigmasq(hp,
                                              low_frequency_cutoff=f) ** 0.5)
        hp *= s
        hp.params = kwds
        hp.view = hp[self.kmin:-1]
        hp.s = (1.0 / s) ** 2.0
        return hp

    def match(self, hp, hc):
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f

r = 0
if not args.tolerance:
    tolerance = (1 - args.minimal_match) / 10
else:
    tolerance = args.tolerance

size = int(1.0 / tolerance)

gen = GenUniformWaveform(args.buffer_length,
    args.sample_rate, args.low_frequency_cutoff)
bank = TriangleBank()

def wf_wrapper(p):
    try:
        hp = gen.generate(**p)
        return hp
    except Exception:
        return None
            
if args.input_file:
    f = h5py.File(args.input_file, 'r')
    params = {k: f[k][:] for k in f}
    bank, _ = bank.check_params(gen, params, args.minimal_match)

# Newtonian estimate of the merger time
# Expression taken from Eq. (12) in https://arxiv.org/pdf/1911.06024
def tmerg(mass1, mass2, e, fmin):
    q = pycbc.conversions.q_from_mass1_mass2(mass1, mass2)
    Mtot = mass1 + mass2

    omega_start = numpy.pi * fmin * Mtot * lal.MTSUN_SI
    # Use Kepler 3rd law
    a0 = omega_start ** (-2 / 3.0)

    e2 = e * e
    e4 = e2 * e2

    f_e = (1.0 + (73.0 / 24) * e2 + (37.0 / 96.0) * e4) / (1 - e2) ** 3.5
    t_merger = 5 * ((1 + q) ** 2) * (a0**4.0) / (256 * q * f_e)
    t_merger_SI = t_merger * Mtot * lal.MTSUN_SI

    return t_merger_SI

def draw(rtype):

    if rtype == 'uniform':
        if args.input_config is None:
            params = {name: numpy.random.uniform(pmin, pmax, size=size)
                      for name, pmin, pmax in zip(args.params, args.min, args.max)}
        else:
            # `draw_samples_from_config` has its own fixed seed, so must overwrite it.
            random_seed = numpy.random.randint(low=0, high=2**32-1)
            samples = draw_samples_from_config(args.input_config, size, random_seed)
            params = {name: samples[name] for name in samples.fieldnames}
            # Add `static_args` back.
            if static_args is not None:
                for k in static_args.keys():
                    params[k] = numpy.array([static_args[k]]*size)

    elif rtype == 'kde':
        trail = 300
        if trail > len(bank):
            trail = len(bank)
        p = bank.keys()
        p = [k for k in p if k not in fdict]
        p.remove('approximant')
        p.remove('f_lower')
        if args.input_config is not None:
            p = variable_args
        bdata = numpy.array([bank.key(k)[-trail:] for k in p])
        kde = gaussian_kde(bdata)
        points = kde.resample(size=size)
        params = {k: v for k, v in zip(p, points)}

        # Add `static_args` back, some transformations may need them.
        if args.input_config is not None and static_args is not None:
            for k in static_args.keys():
                params[k] = numpy.array([static_args[k]]*size)

        # Apply `waveform_transforms` defined in the .ini file to samples.
        if args.input_config is not None and waveform_transforms is not None:
            params = transforms.apply_transforms(params, waveform_transforms)

    params['approximant'] = numpy.array([args.approximant]*size)

    # Filter out stuff (kde method may also generate samples outside boundaries).
    l = None
    if args.input_config is None:
        for name, pmin, pmax in zip(args.params, args.min, args.max):
            nl = (params[name] < pmax) & (params[name] > pmin)
            l = (nl & l) if l is not None else nl

        if args.max_q:
            q =  numpy.maximum(params['mass1'] / params['mass2'], params['mass2'] / params['mass1'])
            l &= q < args.max_q

        if args.max_mtotal:
            l &= params['mass1'] + params['mass2'] < args.max_mtotal

        if args.max_mchirp:
            from pycbc.conversions import mchirp_from_mass1_mass2
            mc = mchirp_from_mass1_mass2(params['mass1'], params['mass2'])
            l &= mc < args.max_mchirp

        if args.min_mchirp:
            from pycbc.conversions import mchirp_from_mass1_mass2
            mc = mchirp_from_mass1_mass2(params['mass1'], params['mass2'])
            l &= mc > args.min_mchirp

    else:
        l = dists_joint.contains(params)

    params = {k: params[k][l] for k in params}
    return params

def cdraw(rtype, ts, te):
    from pycbc.conversions import tau0_from_mass1_mass2

    p = draw(rtype)
    if  len(p[list(p.keys())[0]]) > 0:
        t = tau0_from_mass1_mass2(p['mass1'], p['mass2'],
                                  args.tau0_cutoff_frequency)
        l = (t < te) & (t > ts)
        p = {k: p[k][l] for k in p}

    i = 0
    while len(p[list(p.keys())[0]]) < size:

        tp = draw(rtype)
        p = {k: numpy.concatenate([p[k], tp[k]]) for k in p}

        if  len(p[list(p.keys())[0]]) > 0:
            t = tau0_from_mass1_mass2(p['mass1'], p['mass2'],
                                      args.tau0_cutoff_frequency)
            l = (t < te) & (t > ts)
            p = {k: p[k][l] for k in p}

        i += 1
        if i > args.placement_iterations:
            break

    if len(p[list(p.keys())[0]]) == 0:
        return None

    return p
    
tau0s = args.tau0_start
tau0e = tau0s + args.tau0_crawl
while tau0e <= args.tau0_end:
    accept = 1
    r = 0
    while accept > tolerance:
        # Standard Round
        r += 1
        params = cdraw('uniform', tau0s, tau0e)
        if params is None:
            break

        blen = len(bank)
        bank, uaccept = bank.check_params(gen, params, args.minimal_match)
        logging.info("tau0 %3.1f-%3.1f: uniform(round %s) finished! "
                     "banksize:%s accept:%s added:%s\n",
                     tau0s, tau0e, r, len(bank), uaccept, len(bank) - blen)
        
        # only start to determine the acceptance when going over 10 rounds
        if r > 10:
            accept = uaccept 

        # activate a KDE round after a uniform round
        kloop = 0
        while ((kloop == 0) or (kaccept / okaccept) > .5) and len(bank) > 10:
            r += 1
            kloop += 1
            params = cdraw('kde', tau0s, tau0e)
            blen = len(bank)
            bank, kaccept = bank.check_params(gen, params, args.minimal_match)
            logging.info("tau0 %3.1f-%3.1f: KDE(round %s in total %s) finished! "
                         "banksize: %s accept: %s added: %s\n",
                         tau0s, tau0e, kloop, r, 
                         len(bank), kaccept, len(bank) - blen)
            
            if uaccept:
                logging.info('Ratio of acceptances: %2.3f' % (kaccept / (uaccept)))

            if kloop == 1:
                okaccept = kaccept

            if kaccept <= tolerance:
                accept = kaccept
                break

    bank.culltau0(tau0s - args.tau0_threshold * 2.0)
    logging.info("Region Done %3.1f-%3.1f, %s stored", tau0s, tau0e, bank.activelen())

    tau0s += args.tau0_crawl / 2
    tau0e += args.tau0_crawl / 2

o = h5py.File(args.output_file, 'w')
o.attrs['minimal_match'] = args.minimal_match

for k in bank.keys():
    val = bank.key(k)
    if val.dtype.char == 'U':
        val = val.astype('bytes')
    o[k] = val
