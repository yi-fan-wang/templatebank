from argparse import ArgumentParser
import h5py
import numpy as np


def main():
    parser = ArgumentParser()

    parser.add_argument('--file-prefix', type=str, required=True,
                        help="Common file prefix name to combine")
    parser.add_argument('--file-number', type=int, required=True,
                        help="Number of files to be combined")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to output files")
    args = parser.parse_args()

    data = {}
    for i in range(args.file_number):
        try:
            filename = args.file_prefix+str(i)+'.hdf'
            print('reading: ', filename)
            f = h5py.File(filename, 'r')
        except:
            print("NOT HERE", filename)
            continue

        for k in f:
            if k not in data:
                data[k] = []
            data[k].append(f[k][:])

    for k in data:
        data[k] = np.concatenate(data[k])

    o = h5py.File(args.output, 'w')
    o.attrs['minimal_match'] = f.attrs['minimal_match']
    print('minimal match: ', o.attrs['minimal_match'])
    for k in data:
        o[k] = data[k]


if __name__ == "__main__":
    main()
