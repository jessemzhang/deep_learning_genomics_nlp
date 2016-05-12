from pbcore.io import FastqIO
import json
import sys

# Usage 
# python get_kmer_dict.py <fastq_name> <k> <out_dir_path>

flname = sys.argv[1]
k = int(sys.argv[2])
out_path = sys.argv[3]
dict_kmer = {}

cell_name = flname.split('.')[0]

json_name = out_path+cell_name+'.json'

reader = FastqIO.FastqReader(flname)

for read_no,record in enumerate(reader):
    seq_length = len(record.sequence)
    print read_no
    for i in xrange(seq_length-k+1):
        dict_kmer.setdefault(record.sequence[i:i+k],0)
        dict_kmer[record.sequence[i:i+k]] += 1
    # if read_no==2:
    #     break

with open(json_name, 'w') as outfile:
    json.dump(dict_kmer, outfile)