import numpy as np
import h5py
from argparse import ArgumentParser

from pyseobnr.generate_waveform import GenerateWaveform

from tqdm import tqdm
import multiprocessing
import pandas as pd

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

class GenUniformWaveform(object):
    '''Waveform Generator
    '''
    def __init__(self, buffer_length, sample_rate, f_lower, add_sigma = False):
        self.f_lower = f_lower
        self.delta_f = 1.0 / buffer_length
        tlen = int(buffer_length * sample_rate) # buffer length x sample_rate
        self.flen = tlen // 2 + 1

        #psd is hard coded to O3 psd
        psd = pycbc.psd.read.from_txt('/work/yifanwang/ecc/templatebank/o3psd.txt', 
            self.flen, self.delta_f, self.f_lower, is_asd_file = False)
        
        self.kmin = int(f_lower * buffer_length)
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(np.float32)

    def generate(self, **kwds):  
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f,
                                                   f_lower = self.f_lower,
                                                   **kwds)
        
        hp.resize(self.flen)
        hp = hp.astype(np.complex64)
        
        hp[self.kmin:-1] *= self.w
        s = pycbc.filter.sigmasq(hp,low_frequency_cutoff=self.f_lower)
        hp /= s**0.5 # normalize waveform
        hp.s = s
        return hp

def gens_wfwrapper(p):
    index = p['index']
    try:
        hp = gen.generate(**p)
        return index, hp
    except Exception as e:
        print(e)
        return index, None
    except:
        print("Unknown error")
        return index, None
    

def wf_wrapper(p):
    p2 = {"approximant": "SEOBNRv5EHM",
          "ModeArray": [(2,2)],
          "f22_start": 20,
          "lmax_nyquist": 1
         }
    p.update(p2)
    index = p['index']
    try:
        wf = GenerateWaveform(p)
        hp, _ = wf.generate_td_polarizations()
        return index, abs(float(hp.epoch))
    except Exception as e:
        print(e)
        return index, None

def main():
    parser = ArgumentParser()

    parser.add_argument('--input', type=str, required=True,
                        help="Input file to add template_duration or template_s to.")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output bank with durations.")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization.")
    parser.add_argument('--add-template-duration', action='store_true', help="Add template duration to the bank")
    parser.add_argument('--add-sigma', action='store_true',help="Add sigma to the bank")
    parser.add_argument('--for-template-bank', action='store_true', help="Add for template bank rather than injections")
    args = parser.parse_args()

    p = {} 
    duration_cache = {}
    s_cache = {}

    if args.input.endswith('.csv'):
        # Read the CSV file
        df = pd.read_csv(args.input)
        df['index'] = df.index
        # generate waveforms
        

        if args.add_template_duration:
            param_list = ['index', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
            with multiprocessing.Pool(args.nprocesses) as pool:
                for return_i, return_epoch in pool.imap_unordered(
                    wf_wrapper,
                    ({k: df.loc[idx, k] for k in param_list} for idx in tqdm(df.index))
                ):
                    if return_epoch is not None:
                        duration_cache[return_i] = return_epoch
            df['template_duration'] = df['index'].map(duration_cache)

        if args.add_sigma:
            df['approximant'] = 'SEOBNRv5E'
            global gen
            gen = GenUniformWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)
            if args.for_template_bank:
                param_list = ['index', 'bmass1', 'bmass2', 'bspin1z', 'bspin2z', 'beccentricity', 'brel_anomaly', 'approximant']
                with multiprocessing.Pool(args.nprocesses) as pool:
                    for return_i, return_hp in pool.imap_unordered(
                        gens_wfwrapper,
                        ({k[1:] if k!='index' and k!='approximant' else k: df.loc[idx, k] for k in param_list} for idx in tqdm(df.index))
                    ):
                        if return_hp is not None:
                            s_cache[return_i] = return_hp.s
                df['btemplate_s'] = df['index'].map(s_cache)
            else:
                param_list = ['index', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly', 'approximant']
                with multiprocessing.Pool(args.nprocesses) as pool:
                    for return_i, return_hp in pool.imap_unordered(
                        gens_wfwrapper,
                        ({k: df.loc[idx, k] for k in param_list} for idx in tqdm(df.index))
                    ):
                        if return_hp is not None:
                            s_cache[return_i] = return_hp.s
                df['template_s'] = df['index'].map(s_cache)

        df.drop('index', axis=1, inplace=True)
        df.to_csv(args.output, index=False)

    elif args.input.endswith('.hdf5'):
        with h5py.File(args.input, 'r') as f:
            for k in f.keys():
                p[k] = f[k][:]
        p['index'] = np.arange(len(p['approximant']))

        sorti = np.argsort(list(duration_cache.keys()))
        duration = np.array(list(duration_cache.values()))[sorti]

        with h5py.File(args.output,'w') as f_write:
            with h5py.File(args.input,'r') as f_bank:
                for k in f_bank.keys():
                    if k != 'template_duration':
                        f_write[k] = f_bank[k][()]
            # https://github.com/h5py/h5py/issues/1329
            f_write['template_duration'] = duration

if __name__ == "__main__":
    main()