import csv
import os
import glob
import re

# id, filename, pair_id
def generate_pairings(dir, debug=False):
    with open(os.path.join(dir, 'pairing.csv'), mode='w') as csv_file:
        # returns a list of dapi/trans tuples for a given folder
        input_files = glob.glob(dir + '*.trans.tif')
        num_input_files = len(input_files)
        # print("Num input files: {0}".format(num_input_files))
        pruned_input_files = []
        trans_pattern = re.compile(r"_[A-Za-z0-9]+_\d+ms\.trans\.tif")
        for trans_file in input_files:
            dapi_file = trans_file.replace('trans.tif', 'dapi.tif')
            if os.path.isfile(dapi_file):
                pruned_input_files.append((trans_file, dapi_file))
            else: # next try to remove the exposure time from trans...
                subbed = trans_pattern.sub('*.dapi.tif', trans_file)
                matching_dapis = glob.glob(subbed)
                if len(matching_dapis) > 0:
                    pruned_input_files.append((trans_file, matching_dapis[0]))
        if debug:
            print("Num input pairs in {0}: {1}\n\n".format(os.path.basename(folder_name), len(pruned_input_files)))
            # print("First 2 input file pairs: \n\n{0}\n\n{1}".format(pruned_input_files[0], pruned_input_files[1]))
        return pruned_input_files

        fieldnames = ['', '']


# generates list of filename tuples from p
def get_pairings(dir):
    pairs = []
    with open('employee_birthday.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        id_dict = {}
        headers = next(csv_reader)
        for row in csv_reader:
            id_dict[row[0]] = row
        for id,row in id_dict.items():
            if row[2] is not None:
                pairs.append((row[1], id_dict[row[2]][1]))
    return pairs


generate_pairings()