from pbcore.io import FastqIO
import json
import sys,os
import multiprocessing as mp

# Usage 
# python get_kmer_dict.py  <k> <in_dir_path> <out_dir_path>

k = int(sys.argv[1])
in_path = sys.argv[2]
out_path = sys.argv[3]


def process_file(flname):
	dict_kmer = {}
	cell_name = flname.split('.')[0]

	json_name = out_path+cell_name+'.json'

	reader = FastqIO.FastqReader(in_path+flname)
	print flname
	for read_no,record in enumerate(reader):
	    seq_length = len(record.sequence)
	    # if read_no%100000 == 0:
	    #     print read_no,flname
	    for i in xrange(seq_length-k+1):
	        dict_kmer.setdefault(record.sequence[i:i+k],0)
	        dict_kmer[record.sequence[i:i+k]] += 1
	    # if read_no==2:
	    #     break

	with open(json_name, 'w') as outfile:
	    json.dump(dict_kmer, outfile)

flnames = sorted([x for x in os.listdir(in_path) if x.endswith('.fastq.gz')])
pool=mp.Pool(processes=32)
pool.map(process_file,flnames)
