#
# This example code has been developed using python 3.9 and tensorflow 2.8.0
#
# material order: background (0), internal air, bone, muscle, fat, WM, GM, CSF, eye
#

import os
import tensorflow as tf
from tensorflow import keras
from functions import dice_coef_loss, dice_coef, load_data, test_model

# Initialization
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT available")
path_to_data_dir = os.getcwd() + os.path.sep + "Data"

# Select either Sum-of-Squares or TxRx intensity normalization
file_tags = ['T1w', 'SoS', 'Segmentation']
# file_tags = ['T1w', 'TxRx', 'Segmentation']

# Load data
data_list = load_data(path_to_data_dir, file_tags)

# Load network
model_tra = keras.models.load_model(os.getcwd() + os.path.sep + 'ForkNET_7T_' + file_tags[1] + '_tra.h5', compile=False)
model_sag = keras.models.load_model(os.getcwd() + os.path.sep + 'ForkNET_7T_' + file_tags[1] + '_sag.h5', compile=False)
model_cor = keras.models.load_model(os.getcwd() + os.path.sep + 'ForkNET_7T_' + file_tags[1] + '_cor.h5', compile=False)
model_tra.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
model_sag.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])
model_cor.compile(optimizer='adam', loss=dice_coef_loss, metrics=[dice_coef])

# Apply network
for i_test in range(len(data_list)):
    print('testing on subject #' + str(i_test) + '...')
    test_model(model_tra, model_sag, model_cor, data_list, i_test)


