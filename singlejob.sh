step=2
start=`echo "$step * $1" | bc`
end=`echo "$step * ($1+1)" | bc`
echo "input: " $1 "tau0 region: "$start $end
echo "cpu number: " $2

OMP_NUM_THREADS=1 python /work/yifanwang/ecc/src/templatebank/pycbc_brute_bank_multicore.py \
--verbose \
--output-file eccbank-$1.hdf \
--minimal-match 0.95 \
--tolerance .1 \
--placement-iterations 1000000 \
--buffer-length 4 \
--sample-rate 2048 \
--tau0-threshold 0.5 \
--approximant SEOBNRv5E \
--tau0-crawl 1 \
--tau0-start $start \
--tau0-end  $end \
--max-q 8 \
--psd-file /work/yifanwang/ecc/templatebank/o3psd.txt \
--min 5 5 -0.5 -0.5 0 0 \
--max 100 100 0.5 0.5 0.4 6.283185307179586 \
--params mass1 mass2 spin1z spin2z eccentricity rel_anomaly \
--seed 1 \
--low-frequency-cutoff 20 --nprocesses $2
