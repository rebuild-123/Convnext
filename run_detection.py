#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import cv2
import sys
import json
import shutil
import numpy as np
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from random import random
from datetime import datetime
import matplotlib.pyplot as plt

from model import Model,Convnext_Model
from dataset import data_augmentations,Dataset_2


# In[2]:


model_path = './convnext_aug/eval_miou_0.6777'
# model_path = './results/deeplabv3plus_convnext/convnext_aug/eval_miou:0.6777'


# In[3]:


config_path = './configs/convnext_confing.json'
with open(config_path,'r') as f:
    config = json.load(f)


# In[4]:


# validation_dataset,validation_num = Dataset(
#     imgs_splits_path = sys.argv[0],
#     labs_splits_path = sys.argv[0],
#     batchsize = config['training_settings']['base']['batch_size']
# )


# In[5]:


validation_dataset,validation_num = Dataset_2(
    imgs_splits_path = sys.argv[1],
    labs_splits_path = sys.argv[1],
    batchsize = config['training_settings']['base']['batch_size']
)


# In[6]:


# validation_dataset,validation_num = Dataset(
#     imgs_splits_path = '/workdir/security/home/junjiehuang2468/contests/aidea/dataset/ICME2022_Training_Dataset/splits/validation_imgs.txt',
#     labs_splits_path = '/workdir/security/home/junjiehuang2468/contests/aidea/dataset/ICME2022_Training_Dataset/splits/validation_imgs.txt',
#     batchsize = config['training_settings']['base']['batch_size']
# )


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    tf.saved_model.load(sess,[tf.saved_model.tag_constants.TRAINING],model_path)
    
    x = tf.get_default_graph().get_tensor_by_name('Placeholder:0')
    y = tf.get_default_graph().get_tensor_by_name('Placeholder_1:0')

    x_aug = tf.get_default_graph().get_tensor_by_name('Placeholder_2:0')
    y_aug = tf.get_default_graph().get_tensor_by_name('Placeholder_3:0')

    batch_images_aug,batch_labels_aug = data_augmentations(x,y)
    output = tf.get_default_graph().get_tensor_by_name('unsample:0')
    
    validation_iterator = validation_dataset.make_one_shot_iterator()
    validation_images, validation_labels, validation_filenames = validation_iterator.get_next()
    validation_iter_num = validation_num/config['training_settings']['base']['batch_size']
    tbar = tqdm(range(int(validation_iter_num)+1 if validation_iter_num%1 != 0 else int(validation_iter_num)))
    for i in tbar:
        validation_batch_images,validation_batch_labels, validation_batch_filenames = sess.run([validation_images,validation_labels, validation_filenames])
        validation_batch_images = np.array([cv2.resize(validation_batch_images[i],(1280,720), interpolation=cv2.INTER_LINEAR) for i in range(len(validation_batch_images))])
        img = sess.run(output,feed_dict={x_aug:validation_batch_images,y_aug:validation_batch_images[:,:,:,0]})
        
        for j in range(len(img)):
            temp = np.repeat(np.expand_dims(np.argmax(img[j],axis=-1),axis=-1),repeats=3,axis=-1)
            temp = cv2.resize(temp,(1920, 1080), interpolation=cv2.INTER_NEAREST).astype('uint8')
            plt.imsave(
                sys.argv[2] + f'{validation_batch_filenames[j].decode("utf-8").split("/")[-1][:-4]}.png',
                temp
            )

