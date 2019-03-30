import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from src.processing.train import train_unet

# train_unet('nucleus-initial', dataset='nucleus',
#            num_layers=6, filter_size=3,
#            learn_rate=1e-4, conv_depth=32, epochs=25,
#            records=-1, batch_size=16, activation=A.PReLU,
#            advanced_activations=True, last_activation=A.PReLU)

# data, label = DataLoader.load_testing(records=-1, dataset='nucleus')
# ssim = prediction('nucleus-initial', data, label)