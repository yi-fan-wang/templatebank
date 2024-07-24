import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

import numpy as np
import numpy
import pandas as pd
import h5py
from argparse import ArgumentParser
from pyseobnr.generate_waveform import GenerateWaveform

import pycbc.conversions
import pycbc.distributions
import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import multiprocessing

class GenUniformWaveform(object):
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
        self.w = ((1.0 / psd[self.kmin:-1]) ** 0.5).astype(numpy.float32)
        
        qtilde = pycbc.types.zeros(tlen, numpy.complex64)
        q = pycbc.types.zeros(tlen, numpy.complex64)
        self.qtilde_view = qtilde[self.kmin:self.flen - 1]
        self.ifft = pycbc.fft.IFFT(qtilde, q)
        
        self.md = q._data[-100:]
        self.md2 = q._data[0:100]

    def generate(self, **kwds):  
        if kwds['approximant'] in pycbc.waveform.fd_approximants():  
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f,**kwds)
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
        hp.view = hp[self.kmin:-1]
        hp.s = s

        return hp

    def match(self, hp, hc):
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
        print(e)
        return None

def main():
    parser = ArgumentParser()

    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output bank with waveforms.")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization.")
    args = parser.parse_args()

    # initialize a waveform generator
    global gen
    gen = GenUniformWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    p = {}
    with h5py.File(args.bank,'r') as f:
        for k in f.keys():
            if k == 'approximant':
                p[k] = [f[k][v].decode() for v in range(len(f[k]))]
            else:
                p[k] = f[k][:]
    p['index'] = np.arange(len(p['approximant']))

    # generate waveforms
    waveform_cache = {}
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i, return_hp in pool.imap_unordered(
            wf_wrapper,
            ({k: p[k][idx] for k in p.keys()} for idx in tqdm(range(len(p['approximant']))))
            ):
            waveform_cache[return_i] = [return_hp]

    # dump the waveforms
    for ii in tqdm(waveform_cache.keys()):
        waveform_cache[ii][0].save(args.output,group=str(ii))

if __name__ == "__main__":
    main()
