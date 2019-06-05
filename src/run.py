import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.processing.train import train_unet
from src.processing.predict import prediction
from src.data.loader import DataLoader
import keras.layers.advanced_activations as A

# train_unet('nucleus-4dirs', dataset='0129-2dirs',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, batch_size=16, activation=A.PReLU,
#            advanced_activations=True, last_activation=A.PReLU)

# train_unet('nucleus-all-epochs', dataset='nucleus',
#            num_layers=6, filter_size=3, save_best_only=False,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, batch_size=16, activation=A.PReLU,
#            advanced_activations=True, last_activation=A.PReLU)



# data, label = DataLoader.load_testing(records=-1, dataset='0129-2dirs')
# ssim = prediction('unet_6-3_mse_nucleus-small', data, label)


data, label = DataLoader.load_testing(records=-1, dataset='nucleus')
ssim = prediction('unet_6-3_mse_nucleus-all-epochs', data, label,
                  weights_file='weights_25.h5')
