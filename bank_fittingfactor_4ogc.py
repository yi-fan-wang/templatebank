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
import searchtool

#np.random.seed(0)
#toy_num = 20000

def wf_wrapper(p):
    index = p['index']
    try:
        hp = gen.generate(**p)
        return index, hp
    except Exception as e:
        logging.info("Waveform generation failed for #%i: %s", index, e)
        return index, None

def match_wrapper(p):
    '''A wrapper function to compute match
    '''
    h1 =pycbc.types.FrequencySeries(initial_array=p['h1_data'], delta_f=p['h1_delta_f'])
    h2 =pycbc.types.FrequencySeries(initial_array=p['h2_data'], delta_f=p['h2_delta_f'])
    return p['bank_index'], gen.match(h1, h2)

def allinjmatch_wrapper(p):
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
    mass_lim = (100, 200)
    spin_lim = (-0.5, 0.5)
    ecc_lim = (0, 0.05)
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
    gen = searchtool.GenNormWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    # Read the template bank parameters
    t_start = datetime.datetime.now()
    logging.info("Reading bank...")
    
    global bank_params
    with h5py.File(args.bank) as f:
        bank_params = pd.DataFrame({k: v[:] for k, v in f.items()})
        bank_params['tau0'] = pycbc.conversions.tau0_from_mass1_mass2(f['mass1'][:],f['mass2'][:],15)
        bank_params['approximant'] = 'SEOBNRv5_ROM'
    
    logging.info("Reading bank done in %s", datetime.datetime.now()-t_start)
    # Generate waveforms from the template bank
    
    global bank_waveform
    bank_waveform = {}
    if args.bank_waveform is None:
        logging.info("Generating waveforms from a bank...")
        # generate waveforms
        bank_params['index'] = bank_params.index
        parlist = ['index'] + list(bank_params.columns)
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
    inj_params['approximant'] = 'SEOBNRv5E'
    inj_params['f_lower'] = 20
    # Generate waveforms from the simulated signals
    parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
    logging.info("Generating injection waveforms...")
    global inj_waveform
    inj_waveform = {}
    inj_s = {}
    inj_duration = {}
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_hp in pool.imap_unordered(
            wf_wrapper,
            ({k: inj_params.loc[idx,k] for k in parlist} for idx in tqdm(inj_params.index))
        ):
            inj_waveform[return_i] = return_hp
            try:
                inj_s[return_i] = return_hp.params['template_s']
                inj_duration[return_i] = return_hp.params['template_duration']
            except AttributeError:
                continue
    inj_params['template_s'] = inj_params['index'].map(inj_s)
    inj_params['template_duration'] = inj_params['index'].map(inj_duration)
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
                for cname in list(bank_params.columns):
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
