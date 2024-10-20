import numpy as np
import pandas as pd
import h5py

import pycbc.conversions, pycbc.distributions, pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import datetime
import multiprocessing
import uuid
from argparse import ArgumentParser
import logging

#np.random.seed(0)
#toy_num = 20000
class GenWaveform(object):
    '''Waveform Generator
    '''
    def __init__(self, buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate) # buffer length x sample_rate
        self.flen = tlen // 2 + 1

        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt('/work/yifanwang/ecc/templatebank/o3psd.txt', 
            self.flen, self.delta_f, self.f_lower, is_asd_file = False)
        
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(np.float32)
        
        qtilde = pycbc.types.zeros(tlen, np.complex64) # correlation in Fourier domain
        q = pycbc.types.zeros(tlen, np.complex64) # correlation in time domain
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        
        # the maximum is around 0
        self.md = q._data[-100:]
        self.md2 = q._data[0:100] 

    def generate(self, **kwds):
        '''Return normalized hp
        '''
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f, **kwds)
            if hasattr(hp, 'eob_template_duration'):
                duration = hp.eob_template_duration
        else:
            dt = 1.0 / self.sample_rate
            hp = pycbc.waveform.get_waveform_filter(
                        pycbc.types.zeros(self.flen, dtype=np.complex64),
                        delta_f=self.delta_f,
                        delta_t=dt,
                        f_lower=self.f_lower,
                        **kwds)
        
        hp.resize(self.flen)
        hp = hp.astype(np.complex64)
        
        hp[self.kmin:-1] *= self.w
        s = pycbc.filter.sigmasq(hp, low_frequency_cutoff=self.f_lower)
        hp /= s**0.5 
        
        hp.params = kwds
        #hp.s = s
        hp.params['template_s'] = s
        if duration:
            hp.params['template_duration'] = duration

        return hp

    def match(self, hp, hc):
        hp.view = hp[self.kmin:-1]
        hc.view = hc[self.kmin:-1]
        pycbc.filter.correlate(hp.view, hc.view, self.qtilde_view)
        self.ifft.execute()
        m = max(abs(self.md).max(), abs(self.md2).max())
        return m * 4.0 * self.delta_f

    def overlap(self, hp, hc):
        o = hp.inner(hc)
        return o * 4.0 * self.delta_f

def wf_wrapper(p):
    index = p['index']
    try:
        hp = gen.generate(**p)
        return index, hp
    except Exception:
        return index, None

def match_wrapper(p):
    '''A wrapper function to compute match
    '''
    h1 =pycbc.types.FrequencySeries(initial_array=p['h1_data'], delta_f=p['h1_delta_f'])
    h2 =pycbc.types.FrequencySeries(initial_array=p['h2_data'], delta_f=p['h2_delta_f'])
    return p['bank_index'], gen.match(h1, h2)

#inj_params = None
#inj_waveform = {}
#bank_params = None
#bank_waveform = {}
def allinjmatch_wrapper(p):
    #global inj_params
    #global inj_waveform
    #global bank_params
    #global bank_waveform
    
    hpinj = inj_waveform[p['inj_index']]
    if hpinj == None:
        logging.info("Failed waveform generation in injections for #%i", p['inj_index'])
        return p['inj_index'], None, None
        
    ltau0 = abs(bank_params['tau0']- inj_params.loc[p['inj_index'],'tau0']) < p['args'].tau0_tolerance if p['args'].tau0_tolerance > 0 else True
    lduration = abs(bank_params['template_duration']- hpinj.params['template_duration']) < p['args'].duration_tolerance if p['args'].duration_tolerance > 0 else True
    lsigma = np.maximum(hpinj.params['template_s']/bank_params['template_s'], bank_params['template_s']/hpinj.params['template_s']) < p['args'].sigma_tolerance if p['args'].sigma_tolerance > 0 else True

    if p['args'].toy_num is not None:
        neighbor = toy_i 
    else:
        neighbor = bank_params[ltau0 & lduration & lsigma].index
    logging.info("Number of FF jobs = %i", len(neighbor))

    # do some fitting factor calculations
    maxmatch = 0
    maxindex = None
    for jj in neighbor:
        match = gen.match(hpinj, bank_waveform[jj])
        if match > maxmatch:
            maxmatch = match
            maxindex = jj
    return p['inj_index'], maxindex, maxmatch


def gen_injections():
    mass_lim = (5, 200)
    spin_lim = (-0.5, 0.5)
    ecc_lim = (0, 0.5)
    ano_lim = (0, 2*np.pi)

    uniform_prior = pycbc.distributions.Uniform(
                            mass1=mass_lim,
                            mass2=mass_lim,
                            spin1z=spin_lim,
                            spin2z=spin_lim,
                            eccentricity=ecc_lim,
                            rel_anomaly=ano_lim)

    def _q_lt_8(params):
        return pycbc.conversions.q_from_mass1_mass2(params["mass1"],params["mass2"]) < 8
    def _cull_ecc(params):
        return ((params['eccentricity'] < 0.3) & ((params['mass1'] < 15) | (params['mass2']<15))) | ((params['mass1'] >= 15) & (params['mass2'] >= 15))
    return pycbc.distributions.JointDistribution(["mass1",
                                "mass2",
                                "spin1z",
                                "spin2z",
                                "eccentricity",
                                "rel_anomaly"],
                                uniform_prior,
                                constraints=[_q_lt_8, _cull_ecc])

def main():
    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = ArgumentParser()
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank with only parameters")
    parser.add_argument('--bank-waveform', type=str,
                        help="Template bank with waveform pre-stored")
    parser.add_argument('--ninjections', type=int, required=True,
                        help="Number of injections to compute fitting factors")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization. \
                              If not given then only a single core will be used.")
    parser.add_argument('--tau0-tolerance', type=float, default=0,
                        help="Size to measure the neighbors in the template bank of a particular injection")
    parser.add_argument('--duration-tolerance', type=float, default=0,
                        help='Duration tolerance for the waveform generation')
    parser.add_argument('--sigma-tolerance', type=float, default=0,
                        help='Sigma tolerance for the waveform generation') 
    parser.add_argument('--use-parallel-match', action='store_true',help='Use parallel match calculation')
    parser.add_argument('--output', type=str, default='./fitfac.csv',
                        help="Path to output fitting factors.")
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                        help='Checkpoint interval')
    parser.add_argument('--toy-num', type=int, help='Number of templates in toy models')
    args = parser.parse_args()

    global gen
    gen = GenWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    # Read the template bank parameters
    t_start = datetime.datetime.now()
    logging.info("Reading bank...")
    
    global bank_params
    with h5py.File(args.bank) as f:
        bank_params = pd.DataFrame(
           {'mass1': f['mass1'][:],
            'mass2': f['mass2'][:],
            'tau0': pycbc.conversions.tau0_from_mass1_mass2(f['mass1'][:],f['mass2'][:],15),
            'eccentricity': f['eccentricity'][:],
            'rel_anomaly': f['rel_anomaly'][:],
            'spin1z': f['spin1z'][:],
            'spin2z': f['spin2z'][:],
            'approximant': f['approximant'][:].astype('str'),
            'f_lower': f['f_lower'][:],
            'template_duration': f['template_duration'][:],}
        )
        if args.sigma_tolerance > 0:
            bank_params['template_s'] = f['template_s'][:]
            
    logging.info("Reading bank done in %s", datetime.datetime.now()-t_start)
    # Generate waveforms from the template bank
    
    global bank_waveform
    bank_waveform = {}
    if args.bank_waveform is None:
        logging.info("Generating waveforms from a bank...")
        # generate waveforms
        bank_params['index'] = bank_params.index
        parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
        with multiprocessing.Pool(args.nprocesses) as pool:
            for return_i, return_hp in pool.imap_unordered(
                wf_wrapper,
                ({k: bank_params.loc[idx,k] for k in parlist} for idx in tqdm(bank_params.index))
            ):
                bank_waveform[return_i] = return_hp
    else:
        logging.info("Loading waveforms from a bank...")
        if args.toy_num is not None:
            global toy_i
            toy_i = np.random.choice(range(len(bank_params.index)),
                                     args.toy_num,
                                     replace=False)
            for ii in tqdm(bank_params.index[toy_i]):
                bank_waveform[ii] = pycbc.types.load_frequencyseries(args.bank_waveform, str(ii))
        else:
            for ii in tqdm(bank_params.index):
                bank_waveform[ii] = pycbc.types.load_frequencyseries(args.bank_waveform, str(ii))
    logging.info("Bank waveform generation done")

    global inj_params
    # Generate simulated signals
    inj_params = pd.DataFrame(gen_injections().rvs(args.ninjections))
    inj_params['tau0'] = pycbc.conversions.tau0_from_mass1_mass2(inj_params['mass1'],inj_params['mass2'],15)
    inj_params['index'] = inj_params.index
    inj_params['approximant'] = bank_params['approximant'][0]
    inj_params['f_lower'] = bank_params['f_lower'][0]
    # Generate waveforms from the simulated signals
    parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
    logging.info("Generating injection waveforms...")
    global inj_waveform
    inj_waveform = {}
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_hp in pool.imap_unordered(
            wf_wrapper,
            ({k: inj_params.loc[idx,k] for k in parlist} for idx in tqdm(inj_params.index))
        ):
            inj_waveform[return_i] = return_hp
    logging.info("Injection waveforms generation done")

    # Fitting factor calculations
    all_fitting_factors = []
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_maxindex, return_maxmatch in pool.imap_unordered(
            allinjmatch_wrapper,
            ({'inj_index': ii,
              'args': args} for ii in tqdm(inj_params.index))
        ):
            if return_maxmatch == None:
                continue
            dict_current = {'row': return_i, 'fittingfactor': return_maxmatch}
            if return_maxindex is not None:
                for cname in ['eccentricity', 'mass1', 'mass2', 'rel_anomaly', 'spin1z', 'spin2z', 'tau0','template_duration']:
                    dict_current['b'+cname] = bank_params.loc[return_maxindex, cname]
            all_fitting_factors += [dict_current]

    finalize(all_fitting_factors, inj_params, args.output)

def finalize(all_fitting_factors, inj_params, output):
    params_all_fitting_factor = pd.DataFrame(all_fitting_factors)
    result = inj_params.set_index("index").join(params_all_fitting_factor.set_index('row'),
                                                    how='outer',
                                                    rsuffix='_r')
    result.to_csv(output, index=False)

if __name__ == '__main__':
    main()
