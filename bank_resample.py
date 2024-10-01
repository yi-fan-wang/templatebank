import h5py
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--nsamples', type=int,
                        help="Number of resamples")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output the sub bank")
    parser.add_argument('--min-mass', type=float, default=0, help='mass lower bound to resample the bank.')
    parser.add_argument('--select-all', action='store_true', help='Dump all the banks.')
    args = parser.parse_args()

    thisbank = {}
    with h5py.File(args.bank,'r') as bank:
        l = (bank['mass1'][:] > args.min_mass) & (bank['mass2'][:] > args.min_mass) 
        for k in bank.keys():
            thisbank[k] = bank[k][l]
        
        if args.nsamples:
            out_i = np.random.choice(range(len(thisbank['approximant'])),
                                     args.nsamples,
                                     replace=False)
            out_i = np.sort(out_i)
        elif args.select_all:
            out_i = np.arange(len(thisbank['approximant']))
        else:
            raise ValueError('either give a number of samples or select all')
        
        with h5py.File(args.output,'w') as out:
            for k in bank.attrs.keys():
                out.attrs[k] = bank.attrs[k]
            for k in bank.keys():
                val = thisbank[k][out_i]
                if val.dtype.char == 'U':
                    val = val.astype('bytes')
                out[k] = val

if __name__ == '__main__':
    main()