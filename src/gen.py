import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.nucleus import NucleusDataGenerator

# folders for this training session
base_folder = "/home/ubuntu/Dropbox/Voldman_Group/ML with Cell Images/Images/"

folders = [
    base_folder + "01292018_HUVEC(5x_10x)_edgeAndFlat/HUVECs10x_01292018_flat/",
    base_folder + "01292018_HUVEC(5x_10x)_edgeAndFlat/HUVECs10x_01292018_edge/",
    base_folder + "012318_HUVEC(5x_10x_phaseNnucleus_phase)/HUVEC(10x)_01232018/10x_01232018_2/",
    base_folder + "012318_HUVEC(5x_10x_phaseNnucleus_phase)/HUVEC(10x)_01232018/10x_01232018_1/",
    # base_folder + "",
    # base_folder + "",
    # base_folder + "",
    # base_folder + "",
]

# NucleusDataGenerator.generateImages(
#     set_name='0129-2dirs',
#     input_folder=folders,
#     output_folder='../data/')
NucleusDataGenerator.partitionTrainingAndTestSet(set_name='0129-2dirs')


