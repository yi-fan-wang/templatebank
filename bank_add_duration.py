import numpy as np
import numpy
import pandas as pd
import h5py
from argparse import ArgumentParser

import pycbc.conversions
import pycbc.distributions
import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft
from pyseobnr.generate_waveform import GenerateWaveform

from tqdm import tqdm
import multiprocessing

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

    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output bank with waveforms.")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization.")
    args = parser.parse_args()

    p = {}
    with h5py.File(args.bank,'r') as f:
        for k in f.keys():
            p[k] = f[k][:]
    p['index'] = np.arange(len(p['approximant']))

    # generate waveforms
    duration_cache = {}
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_epoch in pool.imap_unordered(
            wf_wrapper,
            ({k: p[k][idx] for k in p.keys()} for idx in tqdm(range(len(p['approximant']))))
            ):
            duration_cache[return_i] = return_epoch

    sorti = np.argsort(list(duration_cache.keys()))
    duration = np.array(list(duration_cache.values()))[sorti]

    with h5py.File(args.output,'w') as f_write:
        with h5py.File(args.bank,'r') as f_bank:
            for k in f_bank.keys():
                f_write[k] = f_bank[k][()]
        # https://github.com/h5py/h5py/issues/1329
        f_write['duration'] = duration

if __name__ == "__main__":
    main()
