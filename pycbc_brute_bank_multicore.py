"""Generate a bank of templates using a brute force stochastic method.
"""
import numpy, h5py
import logging, argparse, time, sys
from scipy.stats import gaussian_kde

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft, pycbc.conversions
from pycbc.conversions import tau0_from_mass1_mass2
import multiprocessing

#import warnings
#import lal
#import lalsimulation as lalsim

from tqdm import tqdm
import logging.config
#import pyseobnr.models.SEOBNRv5EHM
#import pyseobnr.eob.dynamics.integrate_ode_ecc
#import pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt

class GenUniformWaveform(object):
    """
    A class for generating waveforms.

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

    def __init__(self, buffer_length, sample_rate, f_lower, 
                    psd_path = '/work/yifanwang/ecc/templatebank/o3psd.txt'):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate)
        self.flen = tlen // 2 + 1
        
        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt(psd_path,
                                      self.flen,
                                      self.delta_f,
                                      self.f_lower,
                                      is_asd_file = False)
        
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
            if hasattr(hp, 'eob_template_duration'):
                duration = hp.eob_template_duration
        except Exception as e:
            logging.info("Waveform generation failed: %s", e)
            return None
        
        hp.resize(self.flen)
        hp = hp.astype(numpy.complex64)
        hp[self.kmin:-1] *= self.w
        s = float(1.0 / pycbc.filter.sigmasq(hp,low_frequency_cutoff=self.f_lower) ** 0.5)
        hp *= s
        hp.params = kwds
        hp.view = hp[self.kmin:-1]
        hp.s = (1.0 / s) ** 2.0
        if duration:
            hp.params['template_duration'] = duration
        return hp

    def match(self, hp, hc):
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f

def wf_wrapper(p):
    try:
        hp = gen.generate(**p)
        return hp
    except Exception as e:
        logging.info("Waveform generation failed: %s", e)
        return None
        
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
    """A bank of templates that uses the triangle inequality to estimate
    matches based on prior ones.
    """
    def __init__(self, args):
        self.waveforms = []
        self.tbins = {} # tau0 bins
        self.tau0 = numpy.array([])

        self.tau0_threshold = args.tau0_threshold
        self.tau0_cutoff_frequency = args.tau0_cutoff_frequency
        
        self.sigma_threshold = args.sigma_threshold
        if self.sigma_threshold:
            self.sigma = numpy.array([])
            
        self.template_duration_threshold = args.template_duration_threshold
        if self.template_duration_threshold:
            self.template_duration = numpy.array([])

        self.nprocesses = args.nprocesses
        self.minimal_match = args.minimal_match

    def __len__(self):
        return len(self.waveforms)

    def activelen(self):
        i = 0
        for w in self.waveforms:
            if isinstance(w, pycbc.types.FrequencySeries):
                i += 1
        return i

    def __getitem__(self, index):
        return self.waveforms[index]

    def keys(self):
        return list(self.waveforms[0].params)

    def key(self, k):
        return numpy.array([p.params[k] for p in self.waveforms])

    def range(self):
        if not hasattr(self, 'r'):
            self.r = None
        if self.r is None or len(self.r) != len(self):
            self.r = numpy.arange(0, len(self))
        return self.r

    def culltau0(self, threshold):
        """cull waveforms with tau0 less than threshold"""
        class dumb(object):
            pass
        
        t0 = numpy.array([h.tau0 for h in self])
        cull = numpy.where(t0 < threshold)[0]
        for c in cull:
            d = dumb()
            d.tau0 = self.waveforms[c].tau0
            d.params = self.waveforms[c].params
            d.s = self.waveforms[c].s
            self.waveforms[c] = d

    def check_params(self, params):
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
                hp.num_tried = i + 1
                hp.total_num = total_num
                if hp not in self:
                    num_added += 1
                    self.insert(hp)
            else:
                logging.info("%i/%i Waveform generation failed!", i, total_num)
                continue
        del waveform_cache
        return self, num_added / total_num
    
    def __contains__(self, newhp):
        # Apply tau0 threshold
        newhp.tau0 = pycbc.conversions.tau0_from_mass1_mass2(
                                            newhp.params['mass1'],
                                            newhp.params['mass2'],
                                            self.tau0_cutoff_frequency)
        newhp.tbin = int(newhp.tau0 / self.tau0_threshold)
        if newhp.tbin in self.tbins:
            match_range = numpy.array(self.tbins[newhp.tbin],dtype=int)
            range = numpy.where(abs(self.tau0[match_range] - newhp.tau0) < self.tau0_threshold)[0]
            match_range = match_range[range]
        else:
            match_range = numpy.array([],dtype=int)
        ntau0 = len(match_range)

        # Apply sigmas maximal match.
        if self.sigma_threshold:
            sr = self.sigma[match_range]/newhp.s
            isr = newhp.s/self.sigma[match_range]
            range = numpy.where(numpy.maximum(sr, isr) < self.sigma_threshold)[0]
            match_range = match_range[range]
        nsig = len(match_range)

        # Apply template duration bound
        if self.template_duration_threshold:
            t = newhp.params['template_duration']
            range = numpy.where(abs(self.template_duration[match_range] - t) < self.template_duration_threshold)[0]
            match_range = match_range[range]
        ndur = len(match_range)

        neighbor = Shrinker(match_range)
        maxmatch_matrix = numpy.ones(len(self))
        match_matrix = numpy.array([])
        match_matrix_indices = numpy.array([],dtype=int)
        # Try to do some actual matches
        mmax = 0
        while 1:
            j = neighbor.pop()
            if j is None:
                newhp.maxmatch_matrix_r = match_matrix
                newhp.indices = match_matrix_indices
                logging.info("Add (%i/%i) into the bank. BankSize:%i "
                             "Tau0:%i Sigma:%i Dur:%i, Triangle:%i, MaxMatch:%0.3f"
                              % (newhp.num_tried, newhp.total_num,
                                 len(self), ntau0, nsig, ndur, len(match_matrix_indices), mmax))
                return False

            oldhp = self[j]
            m = gen.match(newhp, oldhp)
            if m > self.minimal_match:
                return True
            if m > mmax:
                mmax = m
            
            match_matrix_indices = numpy.append(match_matrix_indices, j)
            match_matrix = numpy.append(match_matrix, m)
            maxmatch_matrix[j] = m

            # Update bounding match values, apply triangle inequality, consider newhp, oldhp and others
            newhp_other_maxmatch = oldhp.maxmatch_matrix_r - m + 1.10
            update = numpy.where(newhp_other_maxmatch < maxmatch_matrix[oldhp.indices])[0]
            maxmatch_matrix[oldhp.indices[update]] = newhp_other_maxmatch[update]
            # oldhp.indices: absolute index of the waveform in the bank

            # Update where to calculate matches
            skip_threshold = 1 - (1 - self.minimal_match) * 2.0
            neighbor.indices = neighbor.indices[maxmatch_matrix[neighbor.indices] > skip_threshold]
    
    def add_existing_bank(self, params, tau0_start, tau0_end):
        for p in params:
            hp = pycbc.types.FrequencySeries(numpy.zeros(gen.flen, dtype=numpy.complex64))
            hp.params = p
            hp.tau0 = tau0_from_mass1_mass2(p['mass1'], p['mass2'], 15)
            hp.tbin = int(hp.tau0 / self.tau0_threshold)
            self.insert(hp)

    def insert(self, hp):
        self.waveforms.append(hp)
        for b in [hp.tbin - 1, hp.tbin, hp.tbin + 1]:
            if b in self.tbins:
                self.tbins[b].append(len(self)-1)
            else:
                self.tbins[b] = [len(self)-1]
        self.tau0 = numpy.append(self.tau0, hp.tau0)
        if self.sigma_threshold:
            self.sigma = numpy.append(self.sigma, hp.s)
        if self.template_duration_threshold:
            self.template_duration = numpy.append(self.template_duration, hp.params['template_duration'])

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
    if args.ecc_constraint:
        l &= ((params['eccentricity'] < 0.3) & ((params['mass1'] < 15) | (params['mass2']<15))) | ((params['mass1'] >= 15) & (params['mass2'] >= 15))

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

def adjustmass(args, tau0s, tau0e):
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        if name == 'mass1':
            mass1min = pmin
            mass1max = pmax
        elif name == 'mass2':
            mass2min = pmin
            mass2max = pmax
    
    massmin = min(mass1min, mass2min)
    massmax = max(mass1max, mass2max)

    while tau0_from_mass1_mass2(massmin, min(massmin * args.max_q, massmax), args.tau0_cutoff_frequency) > tau0e and massmin < massmax:
        massmin += 0.1
    while tau0_from_mass1_mass2(massmax, max(massmax/args.max_q, massmin), args.tau0_cutoff_frequency) < tau0s and massmax > massmin:
        massmax -= 0.1

    for i, name in enumerate(args.params):
        if name == 'mass1' or name == 'mass2':
            args.min[i] = max(massmin - 1, mass1min)
            args.max[i] = min(massmax + 1, mass1max)
    return args.min, args.max

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
    parser.add_argument('--ecc-constraint', action='store_true', help='ecc constraint')
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
    parser.add_argument('--sigma-threshold', type=float)
    parser.add_argument('--template-duration-threshold', type=float)
    parser.add_argument('--tau0-threshold', type=float, required=True, help='threshold to separate two waveforms')
    parser.add_argument('--placement-iterations', default=1000, type=int, 
        help='Specify the number of attempts the bank should make when placing points. Use this option if the bank fails to place any points.')
    parser.add_argument('--tolerance', type=float, required=True, help='tolerance for acceptance')
    parser.add_argument('--size', type=int, required=True,
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
    # checkpointing
    parser.add_argument('--checkpoint-time', type=float, default=5000, help='checkpoint the bank')
    parser.add_argument('--adjust-mass', action='store_true', help='adjust mass range')
    
    pycbc.psd.insert_psd_option_group(parser)
    args = parser.parse_args()

    for model in ["pyseobnr.models.SEOBNRv5EHM",
                  "pyseobnr.eob.dynamics.integrate_ode_ecc",
                  "pyseobnr.eob.dynamics.initial_conditions_aligned_ecc_opt"]:
        logger = logging.getLogger(model)
        logger.disabled = True
    
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO)
    logger.handlers.clear() # Clear existing handlers
    #logger.propagate = False
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    numpy.random.seed(args.seed)    

    global gen
    gen = GenUniformWaveform(args.buffer_length, args.sample_rate, args.low_frequency_cutoff)
    bank = TriangleBank(args)
    
    mass = {}
    # check if the tau0 range is proper
    for name, pmin, pmax in zip(args.params, args.min, args.max):
        if name == 'mass1':
            mass['mass1'] = [pmin, pmax]
        elif name == 'mass2':
            mass['mass2'] = [pmin, pmax]
    taumax = tau0_from_mass1_mass2(mass['mass1'][0], mass['mass2'][0], args.tau0_cutoff_frequency)
    taumin = tau0_from_mass1_mass2(mass['mass1'][1], mass['mass2'][1], args.tau0_cutoff_frequency)

    tau0s = args.tau0_start
    tau0e = tau0s + args.tau0_crawl
    logging.info("Starting to generate stochastic proposals, initial tau0s: %3.2f, initialtau0e: %3.2f, tau0end: %3.2f", tau0s, tau0e, args.tau0_end)
    while tau0e <= args.tau0_end + 0.00000001:
        logging.info("tau0s, tau0e: %3.2f-%3.2f", tau0s, tau0e)
        
        if args.input_file:
            if tau0s == args.tau0_start:
                taubanks = tau0s - args.tau0_threshold
                taubanke = tau0e + args.tau0_threshold
            else:
                taubanks = taubanke 
                taubanke = tau0e + args.tau0_threshold
            logging.info("Adding the existing bank in tau0 range %3.2f-%3.2f", taubanks, taubanke)
            ilength = len(bank)
            f = h5py.File(args.input_file, 'r')
            t = tau0_from_mass1_mass2(f['mass1'][:], f['mass2'][:], args.tau0_cutoff_frequency)
            l = (t <= taubanke) & (t >= taubanks)
            params = {k: f[k][l] for k in f.keys() if k!= 'f_lower' and k!='s' and k!='template_duration'}
            params['approximant'] = numpy.array([v.decode() for v in params['approximant']])

            if len(tuple(params.values())[0]) > 0:
                logging.info('Adding %s waveforms from the existing bank', len(tuple(params.values())[0]))
                bank, _ = bank.check_params(params)
            f.close()
            logging.info("Existing bank added, banksize: %s, adding: %s", len(bank), len(bank)-ilength)

        if args.adjust_mass:
            args.min, args.max = adjustmass(args, tau0s, tau0e)
        for name, pmin, pmax in zip(args.params, args.min, args.max):
            logging.info("parameter %s: %3.3f-%3.3f", name, pmin, pmax)
        
        accept = 1
        loop = 0
        while accept > args.tolerance and tau0s < taumax and tau0e > taumin:
            current_time = time.time()
            # Standard Round
            loop += 1
            params = cdraw('uniform', tau0s, tau0e, args, bank)
            if params is None:
                break
            
            blen = len(bank)
            bank, uaccept = bank.check_params(params)
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
                
                bank, kaccept = bank.check_params(params)
                if kloop == 1:
                    initial_kaccept = kaccept
                logging.info("tau0 %3.1f-%3.1f: KDE(round %s in total %s) finished! "
                             "banksize: %s accept: %s k0accept: %s, added: %s\n",
                             tau0s, tau0e, kloop, loop, 
                             len(bank), kaccept, initial_kaccept, len(bank) - blen)

                if kaccept <= args.tolerance:
                    accept = kaccept
                    break

                if time.time() - current_time > args.checkpoint_time:
                    logging.info("Checkpointing bank")
                    finalize(args, bank, checkpoint=True)
                    current_time = time.time()

        bank.culltau0(tau0s - args.tau0_threshold * 2.0)
        logging.info("Region Done %3.1f-%3.1f, %s stored", tau0s, tau0e, bank.activelen())

        tau0s += args.tau0_crawl / 2
        tau0e += args.tau0_crawl / 2

    finalize(args, bank)

def finalize(args, bank, checkpoint=False):
    if checkpoint:
        o = h5py.File('checkpoint_'+args.output_file, 'w')
    else:
        o = h5py.File(args.output_file, 'w')
    o.attrs['minimal_match'] = args.minimal_match

    if len(bank) == 0:
        if checkpoint:
            logging.info("No waveforms generated. Checkpointing.")
            return
        else:
            logging.info("No waveforms generated. Exiting.")
            sys.exit()
    else:
        for k in bank.keys():
            val = bank.key(k)
            if val.dtype.char == 'U':
                val = val.astype('bytes')
            o[k] = val
        o['f_lower'] = numpy.array([args.low_frequency_cutoff] * len(bank))

if __name__ == '__main__':
    main()
