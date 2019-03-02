# generator of labeled lymphoma test images
import os, glob
import numpy as np
import scipy
import scipy.misc
from PIL import Image
from libtiff import TIFF

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


class NucleusDataGenerator(object):
    @classmethod
    def partitionTrainingAndTestSet(cls, set_name='nucleus'):
        data_folder = '../../data/'
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


    @classmethod
    def generateImages(cls, set_name='nucleus', stride=100, tile=(192,192),
                      input_folder='../../data/nucleus-raw/',
                      output_folder='../../data/',
                      save_npz=True):
        data = None
        labels = None
        seq = 0

        image_folder = os.path.join(output_folder, set_name)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        input_files = glob.glob(input_folder + '*.trans.tif')
        num_input_files = len(input_files)
        for input_file in input_files:
            # open a phase image

            trans = TIFF.open(input_file).read_image() / 4095.
            dapi = TIFF.open(input_file.replace('trans', 'dapi')).read_image() / 4095.
            # trans = tiff.imread(input_file)
            # dapi = tiff.imread(input_file.replace('trans', 'dapi'))
            input_title = os.path.splitext(os.path.basename(input_file))[0]

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

                        transDestFilename = '{0:05}-trans-{1}-{2}.png'.format(seq,transformation,input_title)
                        dapiDestFilename = '{0:05}-dapi-{1}-{2}.png'.format(seq,transformation,input_title)

                        scipy.misc.imsave(os.path.join(image_folder, transDestFilename), transTile)
                        scipy.misc.imsave(os.path.join(image_folder, dapiDestFilename), dapiTile)

                        # append the raw data to the np tensor
                        data[seq, :] = np.rot90(dapi[st_m:end_m, st_n:end_n], int(rot / 90))
                        labels[seq, :] = np.rot90(trans[st_m:end_m, st_n:end_n], int(rot / 90))
                        seq = seq + 1
        data = data[..., np.newaxis]
        labels = labels[..., np.newaxis]
        if save_npz:
            np.savez(os.path.join(output_folder, set_name + '-all.npz'), data=data, labels=labels)



NucleusDataGenerator.generateImages()
NucleusDataGenerator.partitionTrainingAndTestSet()