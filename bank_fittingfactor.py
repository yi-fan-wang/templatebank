import datetime

import numpy as np
import numpy
import pandas as pd
import h5py

import pycbc.conversions
import pycbc.distributions
import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import multiprocessing
import uuid
from argparse import ArgumentParser
import logging

class GenUniformWaveform(object):
    '''Waveform Generator
    '''
    def __init__(self,buffer_length, sample_rate, f_lower):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate) # buffer length x sample_rate
        self.flen = tlen // 2 + 1

        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt('/work/yifanwang/ecc/templatebank/o3psd.txt', 
            self.flen, self.delta_f, self.f_lower, is_asd_file = False)
        
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(numpy.float32)
        
        qtilde = pycbc.types.zeros(tlen, numpy.complex64)
        q = pycbc.types.zeros(tlen, numpy.complex64)
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        
        self.md = q._data[-100:]
        self.md2 = q._data[0:100]

    def generate(self, **kwds):  
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f, **kwds)
        else:
            dt = 1.0 / self.sample_rate
            hp = pycbc.waveform.get_waveform_filter(
                        pycbc.types.zeros(self.flen, dtype=numpy.complex64),
                        delta_f=self.delta_f,
                        delta_t=dt,
                        f_lower=self.f_lower,
                        **kwds)
        
        hp.resize(self.flen)
        hp = hp.astype(numpy.complex64)
        
        hp[self.kmin:-1] *= self.w
        s = pycbc.filter.sigmasq(hp,low_frequency_cutoff=self.f_lower)
        hp /= s**0.5 
        
        hp.params = kwds
        hp.s = s

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
    except Exception as e:
        return index, None

def match_wrapper(p):
    '''A wrapper function to compute match
    '''
    h1 =pycbc.types.FrequencySeries(initial_array=p['h1_data'], delta_f=p['h1_delta_f'],epoch=p['h1_epoch'])
    h2 =pycbc.types.FrequencySeries(initial_array=p['h2_data'], delta_f=p['h2_delta_f'],epoch=p['h2_epoch'])
    return p['bank_index'], gen.match(h1, h2)


def gen_injections():
    mass_lim = (5, 100)
    spin_lim = (-0.5, 0.5)
    ecc_lim = (0, 0.3)
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

    return pycbc.distributions.JointDistribution(["mass1",
                                "mass2",
                                "spin1z",
                                "spin2z",
                                "eccentricity",
                                "rel_anomaly"],
                                uniform_prior,
                                constraints=[_q_lt_8])

def main():
    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = ArgumentParser()
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank with only parameters")
    parser.add_argument('--bank-waveform', type=str, required=True,
                        help="Template bank with waveform pre-stored")
    parser.add_argument('--ninjections', type=int, required=True,
                        help="Number of injections to compute fitting factors")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization. \
                              If not given then only a single core will be used.")
    parser.add_argument('--tau0-tolerance', type=float, default=1.0,
                        help="Size to measure the neighbors in the template bank with a particular injection")
    parser.add_argument('--output', type=str, default='./',
                        help="Path to output fitting factors.")
    args = parser.parse_args()

    global gen
    gen = GenUniformWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    read_t = datetime.datetime.now()
    with h5py.File(args.bank) as f:
        df_bank = pd.DataFrame(
           {'mass1': f['mass1'][:],
            'mass2': f['mass2'][:],
            'tau0': pycbc.conversions.tau0_from_mass1_mass2(f['mass1'][:],f['mass2'][:],15),
            'eccentricity': f['eccentricity'][:],
            'rel_anomaly': f['rel_anomaly'][:],
            'spin1z': f['spin1z'][:],
            'spin2z': f['spin2z'][:]}
        )
    logging.info("Read bank time: %f", (datetime.datetime.now() - read_t).total_seconds())

    wf_cache = {}
    for ii in df_bank.index[:100]:
        wf_cache[ii] = pycbc.types.load_frequencyseries(args.bank_waveform, str(ii))
    logging.info("Read waveform time: %f", (datetime.datetime.now() - read_t).total_seconds())
    

    inj = gen_injections()
    df_ff = pd.DataFrame(inj.rvs(args.ninjections))
    df_ff['tau0'] = pycbc.conversions.tau0_from_mass1_mass2(df_ff['mass1'],df_ff['mass2'],15)
    #for k in df_ff.columns:
    #    df_ff['b'+str(k)] = ""

    #df_ff['fittingfactor'] = ""
    df_ff['index'] = df_ff.index
    df_ff['approximant'] = 'SEOBNRv5E'
    df_ff['f_lower'] = 20.0

    # +
    inj_cache = {}
    parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']

    wf_wrapper_t = datetime.datetime.now()
    logging.info("Generating injection waveforms...")
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_hp in pool.imap_unordered(
            wf_wrapper,
            ({k: df_ff.loc[idx,k] for k in parlist} for idx in tqdm(df_ff.index))
            ):
            inj_cache[return_i] = return_hp
    logging.info("Injection waveforms generation time: %f", (datetime.datetime.now() - wf_wrapper_t ).total_seconds())

    # do fitting factor calculations
    match_t = datetime.datetime.now()
    all_fitting_factors = []
    for ii in tqdm(df_ff.index):
        calls = []
        hpinj = inj_cache[ii]
        if hpinj == None:
            logging.info("Failed waveform generation in injections for #%i", ii)
            continue

        #neighbor = df_bank[abs(df_bank['tau0']- df_ff.loc[ii,'tau0']) < args.tau0_tolerance].index
        neighbor = range(100)     
        calls += [
                {'bank_index': jj,
                'h1_data': hpinj.data,
                'h1_delta_f': hpinj.delta_f,
                'h1_epoch': hpinj.epoch,
                'h2_data': wf_cache[jj].data,
                'h2_delta_f': wf_cache[jj].delta_f,
                'h2_epoch': wf_cache[jj].epoch} for jj in neighbor
            ]
        #logging.info("Number of jobs = %i", len([elem for row in calls for elem in row]))

        # do some fitting factor calculations
        with multiprocessing.Pool(args.nprocesses) as pool:
            maxmatch = 0
            maxindex = None
            for return_jj, return_match in pool.imap_unordered(
                match_wrapper,
                calls
            ):
                if return_match > maxmatch:
                    maxmatch = return_match
                    maxindex = return_jj

            dict_current = {'row': ii, 'fittingfactor': maxmatch, }
            for cname in ['eccentricity', 'mass1', 'mass2', 'rel_anomaly', 'spin1z', 'spin2z', 'tau0']:
                dict_current['b'+cname] = df_bank.loc[maxindex, cname]

            all_fitting_factors += [
                dict_current
            ]
    logging.info("match_t time: %f", (datetime.datetime.now() - match_t ).total_seconds())

    filename = 'fitfac_'+str(uuid.uuid4())[:6]+'.csv'
    df_all_fitting_factor = pd.DataFrame(all_fitting_factors)
    #result = df_ff.drop(columns={'beccentricity'}).set_index("index").join(df_all_fitting_factor.set_index('row'),
    #                                                                     how='outer',
    #                                                                     rsuffix='_r')
    result = df_ff.set_index("index").join(df_all_fitting_factor.set_index('row'), how='outer', rsuffix='_r')
    result.to_csv(args.output + filename, index=False)

if __name__ == '__main__':
    main()
