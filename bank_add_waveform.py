import logging
import numpy as np
import h5py
from argparse import ArgumentParser

import pycbc.waveform, pycbc.filter, pycbc.types, pycbc.psd, pycbc.fft

from tqdm import tqdm
import multiprocessing
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
            hp, _ = pycbc.waveform.get_fd_waveform(delta_f = self.delta_f,**kwds)
        
        hp.resize(self.flen)
        hp = hp.astype(np.complex64)
        
        hp[self.kmin:-1] *= self.w
        s = pycbc.filter.sigmasq(hp,low_frequency_cutoff=self.f_lower)
        hp /= s**0.5 # normalize waveform

        return hp

def wf_wrapper(p):
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

def main():
    parser = ArgumentParser()

    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output bank with waveforms.")
    parser.add_argument('--nprocesses', type=int, default=1,
                        help="Number of processes to use for waveform generation parallelization.")
    parser.add_argument('--add-sigma', action='store_true',help="Add sigma to the bank")
    parser.add_argument('--num-each-chunk', type=int, help="Number of split for the bank")
    parser.add_argument('--nth-iter', type=int, help="The nth iteration")
    args = parser.parse_args()

    # initialize a waveform generator
    global gen
    gen = GenUniformWaveform(buffer_length = 32, sample_rate = 2048, f_lower = 20)

    logger = logging.getLogger()
    logger.handlers.clear() # Clear existing handlers
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("Reading template bank")
    p = {}
    with h5py.File(args.bank,'r') as f:
        for k in f.keys():
            if k != 'approximant':
                p[k] = f[k][:]
        p['approximant'] = np.array([f['approximant'][0].decode()] * len(p['mass1']))
        p['index'] = np.arange(len(p['approximant']))

    total_num = len(p['approximant'])
    logging.info("Total number of waveforms: %d" % total_num)

    if args.num_each_chunk:
        num_iter = int(total_num // args.num_each_chunk + 1)
        logging.info("Number of waveforms in each chunk: %d, number of iterations: %d" % (args.num_each_chunk, num_iter))

    # generate waveforms
    with multiprocessing.Pool(args.nprocesses) as pool:
        if args.num_each_chunk and args.nth_iter == None:
            for i in range(num_iter):
                waveform_cache = {}
                start = i * args.num_each_chunk
                end = min((i+1) * args.num_each_chunk, total_num)
                for return_i, return_hp in pool.imap_unordered(
                    wf_wrapper,
                    ({k: p[k][idx] for k in p.keys()} for idx in tqdm(range(start, end)))
                ):
                    waveform_cache[return_i] = return_hp
                # write the waveforms to the bank
                for ii in tqdm(waveform_cache.keys()):
                    waveform_cache[ii].save(args.output+'-'+str(i)+'.hdf',group=str(ii))
        elif args.num_each_chunk and args.nth_iter!=None:
            i = args.nth_iter
            waveform_cache = {}
            start = i * args.num_each_chunk
            end = min((i+1) * args.num_each_chunk, total_num)
            logging.info("Generating waveforms for iteration %d, start: %d, end: %d" % (i,start,end))
            
            if start > end:
                print("No more waveforms to generate")
                return
            for return_i, return_hp in pool.imap_unordered(
                wf_wrapper,
                ({k: p[k][idx] for k in p.keys()} for idx in tqdm(range(start, end)))
            ):
                waveform_cache[return_i] = return_hp
            
            logging.info("Writing waveforms to the bank")
            # write the waveforms to the bank
            for ii in tqdm(waveform_cache.keys()):
                waveform_cache[ii].save(args.output+'-'+str(i)+'.hdf',group=str(ii))
        else:
            waveform_cache = {}
            for return_i, return_hp in pool.imap_unordered(
                wf_wrapper,
                ({k: p[k][idx] for k in p.keys()} for idx in tqdm(range(total_num)))
            ):
                waveform_cache[return_i] = return_hp
            # write the waveforms to the bank
            for ii in tqdm(waveform_cache.keys()):
                waveform_cache[ii].save(args.output,group=str(ii))
    '''
    if args.add_sigma:
        # add sigma to the bank
        sorti = np.argsort(list(waveform_cache.keys()))
        s = np.array([h.s for h in waveform_cache.values()])[sorti]
        with h5py.File(args.bank,'a') as f_bank:
            f_bank['template_s'] = s
    '''

if __name__ == "__main__":
    main()
