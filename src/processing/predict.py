import keras.models
import keras.losses
import scipy.misc
import numpy as np
import skimage.measure
import os
from keras_contrib.losses import DSSIMObjective
keras.losses.dssim = DSSIMObjective()
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cmocean
import cmocean.cm


# from keras.utils.generic_utils import get_custom_objects
#
# loss = DSSIMObjective()
# get_custom_objects().update({"dssim": loss})


from src.data.loader import DataLoader
from src.processing.folders import Folders
from PIL import Image
from src.visualization.ssim_plotter import SSIMPlotter

def format_and_save(img_array, output_file, dmin=None, dmax=None, transpose=True):
    img_array = img_array.reshape([192, 192])
    if dmin is not None and dmax is not None:
        img_array = (img_array + dmin) / (dmax + dmin)
    else:
        img_array = img_array / np.max(img_array)
    if transpose:
        img = Image.fromarray(np.transpose(np.uint8(255.0 * img_array)))
    else:
        img = Image.fromarray(np.uint8(255.0 * img_array))
    #print('Norm: {0}, Max: {1}\n'.format(dmin, dmax))
    scipy.misc.imsave(output_file, img)

def prediction(model_name, data, labels, save_err_img = False,
               phase_mapping= False, weights_file='weights.h5',
               transpose=True):
    from src.processing.train import get_unet
    from keras.optimizers import Adam
    model = keras.models.load_model(Folders.models_folder() + model_name + '/' + weights_file)
    mp_folder = Folders.predictions_folder() + model_name + '-n{0}/'.format(data.shape[0])
    os.makedirs(mp_folder, exist_ok=True)
    predictions = model.predict(data, batch_size=32, verbose=0)

    predictions = predictions.astype(np.float64)

    predictions=predictions.reshape([data.shape[0],data.shape[1], data.shape[2]])
    labels=labels.reshape([labels.shape[0],labels.shape[1],labels.shape[2]])
    ssim = np.empty([predictions.shape[0]])
    ms_err = np.empty([predictions.shape[0]])

    for i in range(predictions.shape[0]):
        file_prefix = mp_folder + '{0:05}-'.format(i)
        format_and_save(data[i], file_prefix + 'input.png', transpose=transpose)

        ssim[i] = skimage.measure.compare_ssim(predictions[i], labels[i])

        sq_err = np.square(predictions[i] - labels[i])

        dmin = np.abs(min(np.min(predictions[i]), np.min(labels[i])))
        dmax = np.abs(max(np.max(predictions[i]), np.max(labels[i])))
        format_and_save(predictions[i], file_prefix + 'pred.png',
            dmin, dmax, transpose = transpose)
        format_and_save(labels[i], file_prefix + 'label.png',
            dmin, dmax, transpose = transpose)

        ms_err[i] = np.mean(sq_err)
        format_and_save(predictions[i], file_prefix + 'pred.png',
            dmin, dmax, transpose=transpose)
        format_and_save(labels[i], file_prefix + 'label.png',
            dmin, dmax, transpose=transpose)

        if save_err_img:
            smin, smax = np.min(sq_err), np.max(sq_err)
            format_and_save(sq_err, file_prefix + 'err.png',
                smin, smax, transpose=transpose)

    # calculate and save statistics over SSIM
    header = 'Structural Similarity Indices for {0}\n'.format(model_name)
    header = 'Phase Mapping: {0}\n'.format(phase_mapping)
    header += 'N:     {0}\n'.format(ssim.shape[0])

    header += 'SSIM Statistics :\n --------------\n'
    header += 'Mean:  {0}\n'.format(np.mean(ssim))
    header += 'STDEV: {0}\n'.format(np.std(ssim))
    header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ssim), np.argmin(ssim))
    header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ssim), np.argmax(ssim))
    header += 'MSE Statistics :\n --------------\n'
    header += 'Mean:  {0}\n'.format(np.mean(ms_err))
    header += 'STDEV: {0}\n'.format(np.std(ms_err))
    header += 'MIN:   {0}, Record ({1})\n'.format(np.min(ms_err), np.argmin(ms_err))
    header += 'MAX:   {0}, Record ({1})\n\n'.format(np.max(ms_err), np.argmax(ms_err))

    # add index to ssim
    indexed_ssim_mse = np.transpose(np.vstack((np.arange(ssim.shape[0]), ssim, ms_err)))
    # indexed_ssim_mse = np.array(indexed_ssim_mse, dtype=[("idx", int),  ("SSIM", float), ("MSE", float)])

    np.savetxt(mp_folder + 'stats.txt', indexed_ssim_mse, header=header, fmt="%i %10.5f %10.5f")
    np.savez(mp_folder + 'stats.npz', indexed_ssim_mse)
    SSIMPlotter.save_plot(model_name, ssim)
    return ssim


# data, label = DataLoader.load_testing(records=-1, dataset='nucleus')
# ssim = prediction('nucleus-initial', data, label)