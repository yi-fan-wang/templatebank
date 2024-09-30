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
from pycbc.conversions import tau0_from_mass1_mass2
import multiprocessing

import lal
import lalsimulation as lalsim

class Shrinker(object):
    def __init__(self, data):
        self.data = data

    def pop(self):
        if len(self.data) == 0:
            return None
        l = self.data[-1]
        self.data = self.data[:-1]
        return l

class GenUniformWaveform(object):
    """
    A class for generating uniform waveforms.

    Args:
        buffer_length (int): The length of the buffer.
        sample_rate (float): The sample rate.
        f_lower (float): The lower frequency.

    Attributes:
        f_lower (float): The lower frequency.
        delta_f (float): The frequency resolution.
        flen (int): The length of the frequency array.
        kmin (int): The minimum index of the frequency array.
        w (numpy.ndarray): The square root of the inverse of the power spectral density.
        qtilde_view (numpy.ndarray): The view of the qtilde array.
        ifft (pycbc.fft.IFFT): The inverse Fourier transform object.
        md (numpy.ndarray): The last 100 elements of the q array.
        md2 (numpy.ndarray): The first 100 elements of the q array.

    Methods:
        generate(**kwds): Generates a waveform.
        match(hp, hc): Computes the match between two waveforms.

    """

    def __init__(self, buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate)
        self.flen = tlen // 2 + 1
        
        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt('/work/yifanwang/ecc/templatebank/o3psd.txt', 
            self.flen, self.delta_f, self.f_lower, is_asd_file = False)
        
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(numpy.float32)
        
        # inverse FFT
        qtilde = pycbc.types.zeros(tlen, numpy.complex64)
        q = pycbc.types.zeros(tlen, numpy.complex64)
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        
        self.md = q._data[-100:]
        self.md2 = q._data[0:100]

    def generate(self, **kwds):
        """
        Generates a waveform.

        Args:
            **kwds: Additional keyword arguments.

        Returns:
            pycbc.waveform.Waveform: The generated waveform.

        """
        try:    
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f=self.delta_f,
                                                   f_lower=self.f_lower,
                                                   **kwds)
        except Exception as e:
            logging.info("Waveform generation failed: %s", e)
            return None
        
        hp.resize(self.flen)
        hp = hp.astype(numpy.complex64)
        hp[self.kmin:-1] *= self.w
        s = float(1.0 / pycbc.filter.sigmasq(hp,
                                              low_frequency_cutoff=self.f_lower) ** 0.5)
        hp *= s
        hp.params = kwds
        hp.view = hp[self.kmin:-1]
        hp.s = (1.0 / s) ** 2.0
        return hp

    def match(self, hp, hc):
        """
        Computes the match between two waveforms.

        Args:
            hp (pycbc.waveform.Waveform): The first waveform.
            hc (pycbc.waveform.Waveform): The second waveform.

        Returns:
            float: The match between the two waveforms.

        """
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f

def wf_wrapper(p):
    """
    Wrapper function for generating waveform using the `gen.generate` method.

    Parameters:
    p (dict): A dictionary containing the waveform generation parameters.

    Returns:
    numpy.ndarray or None: The generated waveform if successful, None otherwise.
    """
    try:
        hp = gen.generate(**p)
        return hp
    except Exception:
        return None
        
class TriangleBank(object):
    """A bank of templates that uses the triangle inequality to estimate
    matches based on prior ones.

    Attributes:
        waveforms (list): List of waveform templates.
        tbins (dict): Dictionary mapping time bins to waveform indices.
        enable_sigma_bound (bool): Flag indicating whether to enable sigma match bound.
        tau0_threshold (float): Threshold for tau0.
        nprocesses (int): Number of processes for waveform generation.
    """

    def __init__(self, args, p=None):
        self.waveforms = p if p is not None else []
        self.tbins = {}
        self.enable_sigma_bound = args.enable_sigma_bound
        self.tau0_threshold = args.tau0_threshold
        self.nprocesses = args.nprocesses

    def __len__(self):
        return len(self.waveforms)

    def activelen(self):
        i = 0
        for w in self.waveforms:
            if isinstance(w, pycbc.types.FrequencySeries):
                i += 1
        return i

    def insert(self, hp):
        """
        Inserts a waveform into the search bank.

        Parameters:
        hp (Waveform): The waveform to be inserted.

        Returns:
        None
        """
        self.waveforms.append(hp)

        for b in [hp.tbin - 1, hp.tbin, hp.tbin + 1]:
            if b in self.tbins:
                self.tbins[b].append(len(self)-1)
            else:
                self.tbins[b] = [len(self)-1]

    def __getitem__(self, index):
        return self.waveforms[index]

    def keys(self):
        return list(self.waveforms[0].params)

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
        """
        Check if a waveform is contained in the bank.

        Args:
            newhp (Waveform): The newly added waveform.

        Returns:
            bool: True if the waveform is contained in the bank, False otherwise.
        """
        mmax = 0
        mnum = 0
        # Apply sigmas maximal match.
        if self.enable_sigma_bound:
            matches = self.sigma_match_bound(newhp.s)
            r = self.range()[matches > newhp.threshold]
        else:
            matches = numpy.ones(len(self))
            r = self.range()

        msig = len(r)

        # Apply tau0 threshold
        if self.tau0_threshold:
            newhp.tau0 = pycbc.conversions.tau0_from_mass1_mass2(
                                            newhp.params['mass1'],
                                            newhp.params['mass2'],
                                            15)
            newhp.tbin = int(newhp.tau0 / self.tau0_threshold)

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
            m = gen.match(newhp, oldhp)
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
    
    def add_existing_bank(self, params, tau0_start, tau0_end):
        """
        Add an existing bank to the current bank.

        Args:
            params (dict): A dictionary of parameter values.
            tau0_start (float): The starting value of tau0.
            tau0_end (float): The ending value of tau0.

        Returns:
            None
        """
        for p in params:
            hp = pycbc.types.FrequencySeries(numpy.zeros(gen.flen, dtype=numpy.complex64))
            hp.params = p
            hp.tau0 = tau0_from_mass1_mass2(p['mass1'], p['mass2'], 15)
            hp.tbin = int(hp.tau0 / self.tau0_threshold)
            self.insert(hp)

    def check_params(self, params, threshold):
        """
        Check the parameters and add valid waveforms to the bank.

        Args:
            params (dict): A dictionary of parameter values.
            threshold (float): The threshold value.

        Returns:
            tuple: A self class containing the updated bank and the fraction of waveforms added.
        """
        total_num = len(tuple(params.values())[0])        
        waveform_cache = []
        with multiprocessing.Pool(self.nprocesses) as pool:
            for return_wf in pool.imap_unordered(
                wf_wrapper,
                ({k: params[k][idx] for k in params} for idx in range(total_num))
            ):
                waveform_cache += [return_wf]

        num_added = 0
        for i, hp in enumerate(waveform_cache):
            if hp is not None:
                hp.threshold = threshold
                hp.num_tried = i + 1
                hp.total_num = total_num
                if hp not in self:
                    num_added += 1
                    self.insert(hp)
            else:
                logging.info("%i/%i Waveform generation failed!", i, total_num)
                continue

        return self, num_added / total_num


# Newtonian estimate of the merger time
# Expression taken from Eq. (12) in https://arxiv.org/pdf/1911.06024
def tecc_newtonian(mass1, mass2, e, fmin):
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

def tecc_seobnrv5(mass1, mass2, e, s1z, s2z, fmin):

    return lalsim.SimIMRSEOBNRv5ROMTimeOfFrequency(fmin, 
                    mass1 * lal.MSUN_SI, mass2 * lal.MSUN_SI, s1z, s2z)

def draw(rtype, args, bank):
    '''Generate random parameters in each stochastic proposal
    '''
    if rtype == 'uniform':
        params = {name: numpy.random.uniform(pmin, pmax, size=args.size)
                      for name, pmin, pmax in zip(args.params, args.min, args.max)}

    elif rtype == 'kde':
        trail = 300
        if trail > len(bank):
            trail = len(bank)
        p = bank.keys()
        p.remove('approximant')
        #p.remove('f_lower')
        bdata = numpy.array([bank.key(k)[-trail:] for k in p])
        kde = gaussian_kde(bdata)
        points = kde.resample(size=args.size)
        params = {k: v for k, v in zip(p, points)}

    params['approximant'] = numpy.array([args.approximant] * args.size)
    #params['f_lower'] = numpy.array([args.low_frequency_cutoff] * args.size)

    # Filter out stuff (kde method may also generate samples outside boundaries).
    l = None
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

    params = {k: params[k][l] for k in params}
    return params

def cdraw(rtype, ts, te, args, bank):
    p = draw(rtype, args, bank)
    if  len(p[list(p.keys())[0]]) > 0:
        t = tau0_from_mass1_mass2(p['mass1'], p['mass2'], args.tau0_cutoff_frequency)
        l = (t < te) & (t > ts)
        p = {k: p[k][l] for k in p}

    i = 0
    while len(p[list(p.keys())[0]]) < args.size:
        tp = draw(rtype, args, bank)
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

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    pycbc.add_common_pycbc_options(parser)
    parser.add_argument('--input-file',
        help='Bank to use as a starting point.')
    parser.add_argument('--output-file', required=True,
        help='Output file name for template bank.')

    parser.add_argument('--params',
        help='list of paramaters to use', nargs='+')
    parser.add_argument('--min',
        help='list of the minimum parameter values', nargs='+', type=float)
    parser.add_argument('--max',
        help='list of the maximum parameter values', nargs='+', type=float)
    parser.add_argument('--approximant',  required=True, type=str,
        help='The waveform approximant to place')
    # parameter ranges
    parser.add_argument('--fixed-params', type=str, nargs='*')
    parser.add_argument('--fixed-values', type=float, nargs='*')
    parser.add_argument('--max-mtotal', type=float)
    parser.add_argument('--min-mchirp', type=float, help='minimum chirp mass')
    parser.add_argument('--max-mchirp', type=float, help='maximum chirp mass')
    parser.add_argument('--max-q', type=float, help='maximum mass ratio')
    # proposal generation
    parser.add_argument('--minimal-match', default=0.97, type=float, 
        help='minimal match of SNR due to discreteness of the template bank')
    parser.add_argument('--buffer-length', default=4, type=float,
        help='size of waveform buffer in seconds')
    parser.add_argument('--max-signal-length', type= float, 
        help="When specified, it cuts the maximum length of the waveform model to the lengh provided")
    parser.add_argument('--sample-rate', default=2048, type=float,
        help='sample rate in seconds')
    parser.add_argument('--low-frequency-cutoff', default=20.0, type=float)
    parser.add_argument('--enable-sigma-bound', action='store_true')
    parser.add_argument('--tau0-threshold', type=float, help='threshold to separate two waveforms')
    parser.add_argument('--placement-iterations', default=1000, type=int, 
        help='Specify the number of attempts the bank should make when placing points. Use this option if the bank fails to place any points.')
    parser.add_argument('--tolerance', type=float)
    parser.add_argument('--size', type=int,
        help='Size of waveforms in each stochastic proposal.')
    # tau0 crawling parameters
    parser.add_argument('--tau0-crawl', type=float, help='step length tau0 would proceed')
    parser.add_argument('--tau0-start', type=float, help='starting value for tau0')
    parser.add_argument('--tau0-end', type=float, help='ending value for tau0')
    parser.add_argument('--tau0-cutoff-frequency', type=float, default=15.0)
    # multiprocessing
    parser.add_argument('--nprocesses', type=int, default=1,
        help='Number of processes to use for waveform generation parallelization. If not given then only a single core will be used.')
    parser.add_argument('--seed', type=int, default=0)
    
    pycbc.psd.insert_psd_option_group(parser)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    numpy.random.seed(args.seed)

    global gen
    gen = GenUniformWaveform(args.buffer_length, args.sample_rate, args.low_frequency_cutoff)
    bank = TriangleBank(args)
    if args.input_file:
        f = h5py.File(args.input_file, 'r')
        params = {k: f[k][:] for k in f}
        bank, _ = bank.check_params(gen, params, args.minimal_match)
        f.close()
    
    mass = {}
    # check if the tau0 range is proper
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        if name == 'mass1':
            mass['mass1'] = [pmin, pmax]
        elif name == 'mass2':
            mass['mass2'] = [pmin, pmax]
    taumin = tau0_from_mass1_mass2(mass['mass1'][0], mass['mass2'][0], args.tau0_cutoff_frequency)
    taumax = tau0_from_mass1_mass2(mass['mass1'][1], mass['mass2'][1], args.tau0_cutoff_frequency)

    logging.info("Starting to generate stochastic proposals")
    tau0s = args.tau0_start
    tau0e = tau0s + args.tau0_crawl
    while tau0e <= args.tau0_end:
        accept = 1
        loop = 0
        while accept > args.tolerance and tau0s < taumax and tau0e > taumin:
            # Standard Round
            loop += 1
            params = cdraw('uniform', tau0s, tau0e, args, bank)
            if params is None:
                break

            blen = len(bank)
            bank, uaccept = bank.check_params(params, args.minimal_match)
            logging.info("tau0 %3.1f-%3.1f: uniform(round %s) finished! "
                         "banksize:%s accept:%s added:%s\n",
                         tau0s, tau0e, loop, len(bank), uaccept, len(bank) - blen)

            # only start to determine the acceptance when going over 10 rounds
            if loop > 10:
                accept = uaccept 

            # activate a KDE round after a uniform round
            kloop = 0
            kaccept = 1
            initial_kaccept = 1
            while ((kloop == 0) or (kaccept / initial_kaccept) > .5) and len(bank) > 10:
                loop += 1
                kloop += 1
                params = cdraw('kde', tau0s, tau0e, args, bank)
                blen = len(bank)
                
                bank, kaccept = bank.check_params(params, args.minimal_match)
                if kloop == 1:
                    initial_kaccept = kaccept
                logging.info("tau0 %3.1f-%3.1f: KDE(round %s in total %s) finished! "
                             "banksize: %s accept: %s k0accept: %s, added: %s\n",
                             tau0s, tau0e, kloop, loop, 
                             len(bank), kaccept, initial_kaccept, len(bank) - blen)

                if kaccept <= args.tolerance:
                    accept = kaccept
                    break

        bank.culltau0(tau0s - args.tau0_threshold * 2.0)
        logging.info("Region Done %3.1f-%3.1f, %s stored", tau0s, tau0e, bank.activelen())

        tau0s += args.tau0_crawl / 2
        tau0e += args.tau0_crawl / 2
    
    o = h5py.File(args.output_file, 'w')
    o.attrs['minimal_match'] = args.minimal_match

    if len(bank) == 0:
        logging.info("No waveforms generated. Exiting.")
        return None

    for k in bank.keys():
        val = bank.key(k)
        if val.dtype.char == 'U':
            val = val.astype('bytes')
        o[k] = val
    o['f_lower'] = numpy.array([args.low_frequency_cutoff] * len(bank))

if __name__ == '__main__':
    main()
