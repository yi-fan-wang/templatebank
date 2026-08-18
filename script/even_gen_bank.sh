step=0.5                                                                                                                                                                           
start=`echo "$step * $1" | bc`
end=`echo "$step * ($1 + 1)" | bc`
echo "input: " $1 "tau0 region: "$start $end
echo "cpu number: " $2

#former=`echo "$1*2" | bc`
#latter=`echo "$1*2+2" | bc`
#echo "former: " $former "latter: " $latter

OMP_NUM_THREADS=1 \
python /work/yifanwang/o4a_ecc/src/search_tools/pycbc_brute_bank_multicore.py \
--verbose \
--output-file o4a-eccbank-$1.hdf \
--minimal-match 0.95 \
--placement-iterations 100000000000 \
--tolerance .001 \
--size 10000 \
--buffer-length 32 \
--sample-rate 2048 \
--approximant SEOBNRv5E \
--tau0-threshold 0.5 \
--tau0-crawl 0.5 \
--tau0-start $start \
--tau0-end  $end \
--max-q 8 \
--ecc-constraint \
--spin-constraint \
--psd-file /work/yifanwang/o4a_ecc/src/search_tools/script/aligo_O4low_PSD.txt \
--min 5 5 -0.8 -0.8 0 0 \
--max 200 200 0.8 0.8 0.5 6.283185307179586 \
--params mass1 mass2 spin1z spin2z eccentricity rel_anomaly \
--seed 1 \
--adjust-mass \
--low-frequency-cutoff 20 \
--nprocesses $2
