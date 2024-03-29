# generator of labeled lymphoma test images
import os, glob
import numpy as np
import scipy
import scipy.misc
from PIL import Image
from libtiff import TIFF
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.processing.folders import Folders

class NucleusDataGenerator(object):
    @classmethod
    def partitionTrainingAndTestSet(cls, set_name='nucleus'):
        data_folder = Folders.data_folder()
        data_npz = np.load(data_folder + set_name + '-all.npz')
        all_data, all_labels = data_npz['data'], data_npz['labels']
        np.random.seed(0)
        indices = np.random.permutation(all_data.shape[0])
        test_count = int(np.floor(all_data.shape[0] * 0.2))
        test_idx, training_idx = indices[:test_count], indices[test_count:]
        training_data, test_data = all_data[training_idx, :], all_data[test_idx, :]
        training_labels, test_labels = all_labels[training_idx, :, :], all_labels[test_idx, :, :]
        np.savez(os.path.join(data_folder, set_name + '-training.npz'), data=training_data, labels=training_labels)
        np.savez(os.path.join(data_folder, set_name + '-test.npz'), data=test_data, labels=test_labels)


# sort by last-changed date
#

    @classmethod
    def generateTransDapiPairs(cls, folder_name, debug=False):
        # returns a list of dapi/trans tuples for a given folder
        input_files = glob.glob(folder_name + '*.trans.tif')
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


    @classmethod
    def generateImages(cls, set_name='nucleus', stride=100, tile=(192,192),
                      input_folder='../../data/nucleus-raw/',
                      output_folder='../../data/',
                      save_npz=True, debug=False):
        data = None
        labels = None
        seq = 0

        image_folder = os.path.join(output_folder, set_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        folder_to_id = {}
        input_folder_list = isinstance(input_folder, list)
        input_files = []

        if input_folder_list:
            for idx, f in enumerate(input_folder):
                input_files.extend(NucleusDataGenerator.generateTransDapiPairs(f, debug=debug))
                folder_to_id[f] = idx
            print("\n\n---------\nTotal of {0} Trans/Dapi Pairs Detected Across {1} Folders\n---------\n\n".format(
                  len(input_files), len(input_folder)))
        else:
            input_files = NucleusDataGenerator.generateTransDapiPairs(input_folder, debug=debug)
        num_input_files = len(input_files)

        for trans_file, dapi_file in input_files:
            # open phase image and its dapi counterpart
            trans = TIFF.open(trans_file).read_image() / 4095.
            dapi = TIFF.open(dapi_file).read_image() / 4095.

            input_title = os.path.splitext(os.path.basename(trans_file))[0]
            if input_folder_list:
                dir_id = folder_to_id[os.path.dirname(trans_file) + '/']
                input_title = 'dir{0}_{1}'.format(dir_id, input_title)

            viewDapi = Image.fromarray(np.transpose(np.uint8(255.0 * dapi / np.max(dapi))))
            viewTrans = Image.fromarray(np.transpose(np.uint8(255.0 * trans / np.max(trans))))

            M = trans.shape[0]
            N = trans.shape[1]

            # tile the image
            last_M = int(M - (M % stride))
            last_N = int(N - (N % stride))

            M_count = int(np.floor(M/stride))
            N_count = int(np.floor(N/stride))

            if data is None:
                data = np.zeros((num_input_files * 4 * M_count * N_count, tile[0] , tile[1]))
                labels = np.zeros((num_input_files * 4 * M_count * N_count, tile[0] , tile[1]))
                print("Creating a dataset of shape: {0}".format(data.shape))
                # print("data shape: ")
                # print(data.shape)
                # print('\n')
                # print("labels shape: ")
                # print(labels.shape)

            for rot in range(0, 360, 90):
                for m in range(0, last_M, stride):
                    for n in range(0, last_N, stride):
                        st_m, end_m, st_n, end_n = m, tile[0] + m,  n, tile[1] + n

                        if end_m >= M:
                            st_m, end_m = M - 1 - tile[0], M - 1
                        if end_n >= N:
                            st_n, end_n = N - 1 - tile[1], N - 1

                        crop_mn = [st_m, st_n, end_m, end_n]

                        transTile = viewTrans.crop(crop_mn)
                        dapiTile = viewDapi.crop(crop_mn)

                        transTile = transTile.rotate(rot, resample=Image.BICUBIC)
                        dapiTile = dapiTile.rotate(rot, resample=Image.BICUBIC)

                        transformation = "tm_{0}_tn_{1}_rot_{2}".format(st_m, st_n, rot)

                        transDestFilename = '{0:05}-trans-{1}-{2}.png'.format(seq,transformation,input_title.replace('trans', ''))
                        dapiDestFilename = '{0:05}-dapi-{1}-{2}.png'.format(seq,transformation,input_title.replace('trans', ''))

                        scipy.misc.imsave(os.path.join(image_folder, transDestFilename), transTile)
                        scipy.misc.imsave(os.path.join(image_folder, dapiDestFilename), dapiTile)

                        # append the raw data to the np tensor
                        data[seq, :] = np.rot90(trans[st_m:end_m, st_n:end_n], int(rot / 90))
                        labels[seq, :] = np.rot90(dapi[st_m:end_m, st_n:end_n], int(rot / 90))
                        seq = seq + 1
        data = data[..., np.newaxis]
        labels = labels[..., np.newaxis]
        if save_npz:
            np.savez(os.path.join(output_folder, set_name + '-all.npz'), data=data, labels=labels)



# NucleusDataGenerator.generateImages()
# NucleusDataGenerator.partitionTrainingAndTestSet()