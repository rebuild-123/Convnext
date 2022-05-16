import tensorflow as tf
from random import random

def norm_image(image):
    image_decoded = image/255.0
    return image_decoded

def random_brightness(image,label,max_delta=0.4):
    image = tf.image.random_brightness(image,max_delta)
    return image,label

def random_contrast(image,label,max_delta=0.4):
    image = tf.image.random_contrast(image,1-max_delta,1+max_delta)
    return image,label

def random_hue(image,label,max_delta=0.4):
    image = tf.image.random_hue(image,max_delta)
    return image,label

def random_saturation(image,label,max_delta=0.4):
    image = tf.image.random_saturation(image,1-max_delta,1+max_delta)
    return image,label

def random_flip_left_right(image,label,prob=0.5):
    if random() > prob:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)
    return image,label

def random_shift(image,label,shift_percentage=[0.25,0.25]):
    height, width = tf.cast(tf.shape(image)[1],tf.float32),tf.cast(tf.shape(image)[2],tf.float32)
    shift_percentage = tf.convert_to_tensor([i*(random()-0.5) for i in shift_percentage],dtype=tf.float32)
    shift_height = tf.floor(height*shift_percentage[0])
    shift_width = tf.floor(width*shift_percentage[1])
    
    offset_height = tf.cast(tf.math.maximum(tf.constant(0,tf.float32),shift_height),tf.int32)
    offset_width = tf.cast(tf.math.maximum(tf.constant(0,tf.float32),shift_width),tf.int32)
    target_height = tf.cast(height - tf.math.abs(shift_height),tf.int32)
    target_width = tf.cast(width - tf.math.abs(shift_width),tf.int32)
    height,width = tf.cast(height,tf.int32),tf.cast(width,tf.int32)
    
    image = tf.image.crop_to_bounding_box(image,offset_height,offset_width,target_height,target_width)
    label = tf.image.crop_to_bounding_box(label,offset_height,offset_width,target_height,target_width)
    image = tf.image.pad_to_bounding_box(image,offset_height,offset_width,height,width)
    label = tf.image.pad_to_bounding_box(label,offset_height,offset_width,height,width)
    return image,label

def data_augmentations(
    image,label,brightness_max_delta=0.4,contrast_max_delta=0.4,hue_max_delta=0.4,
    saturation_max_delta=0.4,flip_prob=0.5,shift_percentage=[0.25,0.25]
):
    image,label = random_brightness(image,label,max_delta=brightness_max_delta)
    image,label = random_contrast(image,label,max_delta=contrast_max_delta)
    image,label = random_hue(image,label,max_delta=hue_max_delta)
    image,label = random_saturation(image,label,max_delta=saturation_max_delta)
    image,label = random_flip_left_right(image,label,prob=flip_prob)
    image,label = random_shift(image,label,shift_percentage=shift_percentage)
    return image,label