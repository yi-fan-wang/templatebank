import numpy as np
import pandas as pd
import h5py

import pycbc.conversions, pycbc.distributions, pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import multiprocessing
from argparse import ArgumentParser
import logging

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
        q = pycbc.types.zeros(tlen, np.complex64) # correlation
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
    
def failed_wf(p):
    index = p['index']
    try:
        _ = gen.generate(**p)
        return None
    except Exception:
        return index

def main():
    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    
    parser = ArgumentParser(description='Remove failed waveform from the hdf file.')
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank with only parameters")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization. \
                              If not given then only a single core will be used.")
    parser.add_argument('--with-duration', action='store_true',help="If the bank has template_duration.")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output the new file.")
    args = parser.parse_args()

    with h5py.File(args.bank) as f:
        df_bank = pd.DataFrame(
           {'mass1': f['mass1'][:],
            'mass2': f['mass2'][:],
            'eccentricity': f['eccentricity'][:],
            'rel_anomaly': f['rel_anomaly'][:],
            'spin1z': f['spin1z'][:],
            'spin2z': f['spin2z'][:],
            'approximant': f['approximant'][()].astype('str'),
            'f_lower': f['f_lower'][:]}
        )
        if args.with_duration:
            df_bank['template_duration'] = f['template_duration'][:]

    df_bank['index'] = df_bank.index

    global gen
    gen = GenWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    failed_index = []
    parlist = ['index', 'approximant', 'f_lower', 'mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
    with multiprocessing.Pool(args.nprocesses) as pool:
        for return_i in pool.imap_unordered(
            failed_wf,
            ({k: df_bank.loc[idx, k] for k in parlist} for idx in tqdm(df_bank.index))
        ):
            failed_index += [return_i]

    failed_index = [i for i in failed_index if i != None]
    logging.info("%i failed waveforms are at index %s", len(failed_index), failed_index)
    update_df = df_bank.drop(failed_index)

    with h5py.File(args.output,'w') as f_write:
        with h5py.File(args.bank,'r') as f_bank:
            for k in f_bank.keys():
                if k=='approximant':
                    f_write[k] = update_df[k].values.astype('bytes')
                else:
                    f_write[k] = update_df[k].values
            # filling in the attributes
            for k in f_bank.attrs.keys():
                f_write.attrs[k] = f_bank.attrs[k]
    logging.info("The new bank has been saved to %s", args.output)


if __name__ == "__main__":
    main()
