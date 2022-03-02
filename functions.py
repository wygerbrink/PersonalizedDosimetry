import numpy as np
import SimpleITK as sitk
import os
from os import listdir
from os.path import isfile, join
import collections
from tensorflow.keras import backend as K

def load_data(path_to_data_dir, tags):
    '''
    #load all image data and properties from the data directory
    :param path_to_data_dir: path to data directory
    :param tags: file tags for the data
    :return: list of image data and properties including:
    im_path (image path), pr_path (output data path),  im (image data), voxel_size (voxel size), 
    origin (origin of the volume), direction (direction of the volume)
    '''
    data_path_list = [join(path_to_data_dir, f)
                     for f in listdir(path_to_data_dir)
                     if (not (isfile(join(path_to_data_dir, f))))]
    data_path_list.sort()
    
    data_list = []
    node = collections.namedtuple('node', 'im_path pr_path im voxel_size origin direction')
    for data_path in data_path_list:
        im_path = data_path + os.path.sep + data_path[-10:] + ' ' + tags[0] + ' ' + tags[1] + '.nii.gz'
        pr_path = data_path + os.path.sep + data_path[-10:] + ' ' + tags[2] + ' ' + tags[1] + '.nii.gz'
        im, voxel_size, origin, direction = load_im({'im': im_path})
        n = node(im_path=im_path, pr_path=pr_path, im=im, voxel_size=voxel_size, origin=origin, direction=direction)
        data_list.append(n)
    return data_list

def load_im(path):
    '''
    #load image data from the image file
    :param path: image file path
    :return: image and corresponding properties
    '''
    im = sitk.ReadImage(path['im'])
    voxel_size = im.GetSpacing()
    origin = im.GetOrigin()
    direction = im.GetDirection()
    
    # load image data and rescale to zero mean and unit variance in range [0-1]
    im = sitk.GetArrayFromImage(im)
    im = np.float32(im.reshape(im.shape[0],im.shape[1],im.shape[2],1))
    im = (im - im.mean())/im.std()
    im = (im - im.min())/(im.max() - im.min())
    
    return im, voxel_size, origin, direction

def test_model(model_tra, model_sag, model_cor, data_list, i_test=None):
    '''
    #apply model to data and store resulting segmentation
    :param model: keras model of the network
    :param data_list: list of image data and properties
    :param i_test: index of test subject
    '''

    if i_test is None:
        i_test = np.arange(len(data_list))
    else:
        i_test = np.array([i_test])
    
    for i in i_test:
        pve = []
        
        if bool(model_tra): # combine network orientations when provided
            pve = np.asarray(model_tra.predict(data_list[i].im, batch_size=10, verbose=0))
        if bool(model_sag):
            pve = pve + np.moveaxis(np.asarray(model_sag.predict(np.moveaxis(data_list[i].im,2,0), batch_size=10, verbose=0)),1,3)
        if bool(model_cor):
            pve = pve + np.moveaxis(np.asarray(model_cor.predict(np.moveaxis(data_list[i].im,1,0), batch_size=10, verbose=0)),1,2)
        
        pred = np.argmax(pve, axis=0)
        pred = fill_voids(pred, pve, iter=10)[0]
        save_prediction(pred,data_list[i])

def fill_voids(pred, pve, iter):
    '''
    #fill voids in output by propagating a majority vote from neighbouring voxels
    :param pred: network generated segmentation of the input data
    :param pve: raw network output
    :param iter: number of iterations, proportional to the size of the neighbourhood
    :return: segmentation with voids filled
    '''
    mask_voids = (pve[0]<0.025)&(pred==0)
    mask_voids[0:75,:,:] = False    # ignore voids below skull base
    
    if np.count_nonzero(mask_voids):
        if iter>0:
            if iter==10:
                print(" filling " + str(np.count_nonzero(mask_voids)) + " voids...")
            
            # determine the neighboring majority vote
            pve_smooth = 0.25*pve + 0.125*np.roll(pve,[0, 1, 0, 0, 0]) + 0.125*np.roll(pve,[0, -1, 0, 0, 0]) + 0.125*np.roll(pve,[0, 0, 1, 0, 0]) + 0.125*np.roll(pve,[0, 0, -1, 0, 0]) + 0.125*np.roll(pve,[0, 0, 0, 1, 0]) + 0.125*np.roll(pve,[0, 0, 0, -1, 0])
            pve_smooth[0] = pve[0]    # preserve the background channel
            pred[mask_voids] = np.argmax(pve_smooth, axis=0)[mask_voids]
            
            pred, pve = fill_voids(pred, pve_smooth, iter-1)
        else:
            # if maximum number of iterations has been reached, fill with muscle
            print(" filling " + str(np.count_nonzero(mask_voids)) + " voids with muscle...")
            pve[3] = pve[3]*(1-mask_voids) + mask_voids
    else: print(" finished filling voids... ")
        
    return pred, pve

def save_prediction(pred, data_node):
    '''
    #store segmentations in current node of the data directory
    :param pred: network generated segmentation
    :param data_node: current node of the data directory list
    '''
    pred_out = sitk.GetImageFromArray(np.int16(pred))
    pred_out.SetSpacing(data_node.voxel_size)
    pred_out.SetOrigin(data_node.origin)
    pred_out.SetDirection(data_node.direction)
    sitk.WriteImage(pred_out, data_node.pr_path)

def dice_coef_loss(y_true, y_pred):
    '''
    #evaluate dice coefficient loss function
    :param y_true: ground truth image
    :param y_pred: predicted image
    :return: scalar dice coefficient loss
    '''
    return 1 - dice_coef(y_true,y_pred)

def dice_coef(y_true, y_pred, reg=0):
    '''
    #evaluate dice coefficient
    :param y_true: ground truth image
    :param y_pred: predicted image
    :param reg: regularization parameter (optional)
    :return: scalar dice coefficient
    '''
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + reg) / (K.sum(y_true) + K.sum(y_pred) + reg)



