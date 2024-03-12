import argparse
import csv
import re

parser = argparse.ArgumentParser(description='Process some files.')

parser.add_argument('infiles', type=str, nargs='+', help='The input file paths')

args = parser.parse_args()

for infile in args.infiles:
    outfile = re.sub(r"\.fasta", ".csv", infile) 
    with open(infile, "r") as in_f, open(outfile, "w", newline='') as out_f:
        writer = csv.writer(out_f)
        writer.writerow(["title", "sequence", "label"]) 
        for line in in_f:
            items = line.rstrip().split("|")
            if len(items) == 2:  
                title = items[0]
                label = items[1]
                sequence = next(in_f).rstrip()  
                writer.writerow([title, sequence, label])  