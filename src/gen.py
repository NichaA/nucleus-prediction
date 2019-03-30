import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.nucleus import NucleusDataGenerator

# folders for this training session
base_folder = "/home/ubuntu/Dropbox/Voldman_Group/ML\\ with\\ Cell\\ Images/Images/"

folders = [
    base_folder + "01292018_HUVEC\\(5x_10x\\)_edgeAndFlat/HUVECs10x_01292018_flat/",
    base_folder + "01292018_HUVEC\\(5x_10x\\)_edgeAndFlat/HUVECs10x_01292018_edge/",
]
#path_to_dir = path_to_dir.replace(" ", "\\ ")


NucleusDataGenerator.generateImages(
    set_name='0129-2dirs',
    input_folder=folders,
    output_folder='../data/')

NucleusDataGenerator.partitionTrainingAndTestSet()
