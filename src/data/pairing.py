import csv
import os
import glob
import re
import ntpath

PAIRING_FILE = 'pairing.csv'

# id, filename, pair_id
def generate_pairings(dir, debug=False):
    with open(os.path.join(dir, PAIRING_FILE), mode='w', newline='') as csv_file:
        # returns a list of dapi/trans tuples for a given folder
        input_files = glob.glob(dir + '*.trans.tif')
        # num_input_files = len(input_files)
        # print("Num input files: {0}".format(num_input_files))
        pruned_input_files = []
        trans_pattern = re.compile(r"_[A-Za-z0-9]+_\d+ms\.trans\.tif")
        for trans_file in input_files:
            dapi_file = trans_file.replace('trans.tif', 'dapi.tif')
            if os.path.isfile(dapi_file):
                pruned_input_files.append((trans_file, dapi_file))
            else: # go backwards until we find the first metching dapi
                for i in range(len(dapi_file),-1,-1):
                    subbed = dapi_file[:i] + '*.dapi.tif' #trans_pattern.sub('*.dapi.tif', trans_file)
                    matching_dapis = glob.glob(subbed)
                    if len(matching_dapis) > 0:
                        pruned_input_files.append((trans_file, matching_dapis[0]))
                        break;
        # serialize the pairs to pairings.csv
        id = 0
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'filename', 'pair_id'])
        seen_files = set()
        for trans,dapi in pruned_input_files:
            csv_writer.writerow([id, ntpath.basename(trans), id+1])
            csv_writer.writerow([id+1, ntpath.basename(dapi), None])
            seen_files.add(ntpath.basename(trans))
            seen_files.add(ntpath.basename(dapi))
            id = id + 2

        # write the unpaired files:
        for file in glob.glob(dir + '*.*'):
            if file not in seen_files and ntpath.basename(file) != PAIRING_FILE:
                csv_writer.writerow([id, ntpath.basename(file), None])
                id = id + 1

        if debug:
            print("Num input pairs in {0}: {1}\n\n".format(os.path.basename(dir), len(pruned_input_files)))
            # print("First 2 input file pairs: \n\n{0}\n\n{1}".format(pruned_input_files[0], pruned_input_files[1]))
        return pruned_input_files


# generates list of filename tuples from paring.csv
def get_pairings(dir):
    pairs = []
    with open(os.path.join(dir, PAIRING_FILE),'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        id_dict = {}
        headers = next(csv_reader)
        for row in csv_reader:
            id_dict[row[0]] = row
        for id,row in id_dict.items():
            if row[2]:
                pairs.append((row[1], id_dict[row[2]][1]))
    return pairs
