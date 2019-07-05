import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.processing.train import train_unet
from src.losses.avg import mse_ssim_loss
from src.processing.predict import prediction
from src.data.loader import DataLoader
import keras.layers.advanced_activations as A


# Run 100 Epochs
# train_unet('nucleus-4dirs', dataset='0129-2dirs',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=100,
#            records=-1, batch_size=16, activation=A.PReLU,
#            advanced_activations=True, last_activation=A.PReLU)


# Run 25 epochs, saving every epoch's weights
# train_unet('nucleus-25-epochs', dataset='nucleus',
#             num_layers=6, filter_size=3, save_best_only=False,
#             learn_rate=1e-4, conv_depth=32, epochs=25,
#             records=-1, batch_size=16, activation=A.PReLU,
#             advanced_activations=True, last_activation=A.PReLU)


# run 25 epochs with custom loss function
train_unet('nucleus-custom-loss', dataset='0129-2dirs',
           num_layers=6, filter_size=3, loss=mse_ssim_loss(),
           learn_rate=1e-4, conv_depth=32, epochs=25,
           records=-1, batch_size=16, activation=A.PReLU,
           advanced_activations=True, last_activation=A.PReLU)

# run test predictions on existing models
# data, label = DataLoader.load_testing(records=-1, dataset='0129-2dirs')
# ssim = prediction('unet_6-3_mse_nucleus-4dirs', data, label)
