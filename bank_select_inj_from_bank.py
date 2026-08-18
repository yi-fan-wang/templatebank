import h5py
import numpy as np
from argparse import ArgumentParser
import numpy as np

def main():
    parser = ArgumentParser()
    parser.add_argument('--bank', type=str, required=True,
                        help="Template bank")
    parser.add_argument('--inj', type=str, required=True,
                        help="Injection files")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output the new injection files")
    args = parser.parse_args()

    paralist = ['mass1', 'mass2', 'spin1z', 'spin2z', 'eccentricity', 'rel_anomaly']
    with h5py.File(args.bank,'r') as bank:
        with h5py.File(args.inj,'r') as inj:
            nbank = len(bank['mass1'])
            ninj = len(inj['mass1'])
            resample_i = np.random.choice(range(nbank),
                                     ninj,
                                     replace=True)      
            with h5py.File(args.output,'w') as out:
                for k in inj.keys():
                    if k in paralist:
                        out[k] = np.take(bank[k], resample_i)
                    else:
                        out[k] = inj[k][:]
                for k in inj.attrs.keys():
                    out.attrs[k] = inj.attrs[k]
if __name__ == '__main__':
    main()