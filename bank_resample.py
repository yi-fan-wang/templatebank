import h5py
import numpy as np
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--nsamples', type=int, required=True,
                        help="Number of resamples")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output the sub bank")
    args = parser.parse_args()

    with h5py.File(args.bank,'r') as bank: 
        out_i = np.random.choice(range(len(bank['approximant'])),
                                     args.nsamples,
                                     replace=False)
        out_i = np.sort(out_i)
        with h5py.File(args.output,'w') as out:
            for k in bank.attrs.keys():
                out.attrs[k] = bank.attrs[k]
            for k in bank.keys():
                val = bank[k][out_i]
                if val.dtype.char == 'U':
                    val = val.astype('bytes')
                out[k] = val

if __name__ == '__main__':
    main()