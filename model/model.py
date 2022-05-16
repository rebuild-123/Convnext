import numpy as np
import tensorflow as tf

from .utils import *

def Resnet(x,residual_list=[2,2,6,2],is_training=True,reuse=False,label_dim=6,ch = 16):
    with tf.variable_scope('network',reuse=reuse):
        residual_block = bottle_resblock
        
        x = conv(x, channels=ch, kernel=3, strides=1, scope='conv')
        for i in range(residual_list[0]):
            x = residual_block(x,channels=ch,is_training=is_training,downsample=False,scope=f'resblock0_{str(i)}')
        
        x = residual_block(x,channels=ch*2,is_training=is_training,downsample=True,scope='resblock1_0')
        for i in range(1,residual_list[1]):
            x = residual_block(x,channels=ch*2,is_training=is_training,downsample=False,scope=f'resblock1_{str(i)}')
        low_level_features = x
        
        x = residual_block(x,channels=ch*4,is_training=is_training,downsample=True,scope='resblock2_0')
        for i in range(1,residual_list[2]):
            x = residual_block(x,channels=ch*4,is_training=is_training,downsample=False,scope=f'resblock2_{str(i)}')
            
        x = residual_block(x,channels=ch*8,is_training=is_training,downsample=True,scope='resblock3_0')
        for i in range(1,residual_list[3]):
            x = residual_block(x,channels=ch*8,is_training=is_training,downsample=False,scope=f'resblock3_{str(i)}')
        
        # x = batch_norm(x,is_training,scope='batch_norm')
        # x = tf.nn.relu(x)
        # x = tf.reduce_mean(x,axis=[1,2],keepdims=True)
        # x = fully_connected(x,units=label_dim,scope='logit')
    
        return x,low_level_features
    
def Aspp(x,low_level_features_size,channels=512,reuse=False):
    with tf.variable_scope('aspp',reuse=reuse):
        input_size = tf.shape(x)[1:3]
        conv_1x1 = conv(x,channels=channels,kernel=1,strides=1,scope='conv_1x1')
        conv_3x3_1 = conv(x,channels=channels,kernel=3,strides=1,dilation_rate=6,scope='conv_3x3_1')
        conv_3x3_2 = conv(x,channels=channels,kernel=3,strides=1,dilation_rate=12,scope='conv_3x3_2')
        conv_3x3_3 = conv(x,channels=channels,kernel=3,strides=1,dilation_rate=18,scope='conv_3x3_3')
        
        with tf.variable_scope('image_level_features'):
            image_level_features = tf.reduce_mean(x,[1,2],name='global_average_pooling',keep_dims=True)
            image_level_features = conv(image_level_features,channels=channels,kernel=1,strides=1,scope='conv_1x1')
            image_level_features = tf.image.resize_bilinear(image_level_features,input_size,name='upsample_0')
        
        output = tf.concat([conv_1x1,conv_3x3_1,conv_3x3_2,conv_3x3_3,image_level_features],axis=3,name='concat')
        output = conv(output,channels=channels,kernel=1,strides=1,scope='conv_1x1_concat')
        output = tf.image.resize_bilinear(output,low_level_features_size,name='upsample_1')
        
        return output
    
def Left_Part(middle,low_level_features,channels,num_classes,reuse=False):
    with tf.variable_scope('left_part',reuse=reuse):
        low_level_features = conv(low_level_features,channels=channels,kernel=1,strides=1,scope='conv_1x1')
        output = tf.concat([middle,low_level_features],axis=3,name='concat')
        output = conv(output,channels=channels,kernel=3,strides=1,scope='conv_3x3_1')
        output = conv(output,channels=num_classes,kernel=1,strides=1,scope='conv_3x3_2')
        return output
    
def Model(x,channels=512,num_classes=6,residual_list=[2,2,6,2],residual_ch=16,is_training=True,reuse=False):
    input_size = tf.shape(x)[1:3]
    resnet_output,low_level_features = Resnet(
        x,residual_list=residual_list,ch=residual_ch,is_training=is_training,reuse=reuse
    )
    middle = Aspp(
        resnet_output,low_level_features_size=[input_size[0]//2,input_size[1]//2],
        channels=channels,reuse=reuse
    )
    output = Left_Part(middle,low_level_features,channels,num_classes,reuse=reuse)
    return tf.image.resize_bilinear(output,input_size,name='unsample')

def Convnext(x,depths,dims,drop_path_rate=0.2,layer_scale_init_value=1e-6,is_training=True,scope='convnext'):
    dp_rates=[x for x in np.linspace(0, drop_path_rate, sum(depths))] 
    with tf.variable_scope(scope):
        cur = 0
        for i in range(4):
            x = downsample_layer(x,dim=dims[i],scope=f'downsample_layer_{i}') if i != 0 else stem(x,dims[i])
            for j in range(depths[i]):
                x = block(
                    x,dim=dims[i],
                    drop_prob=dp_rates[cur+j],
                    layer_scale_init_value=layer_scale_init_value,
                    is_training=is_training,
                    scope=f'inverted_bottleneck_block_{i}_{j}'
                )
            if i == 0: low_level_features = x
            cur += depths[i]
        return layer_norm(x),low_level_features
    
def Convnext_Model(x,channels=512,num_classes=6,depths = [3,3,9,3],dims = [96,192,384,768], is_training=True,reuse=False,drop_path_rate=0.2):
    input_size = tf.shape(x)[1:3]
    convnext_output,low_level_features = Convnext(x,depths=depths,dims=dims, drop_path_rate=drop_path_rate)
    middle = Aspp(
        convnext_output,low_level_features_size=[input_size[0]//2,input_size[1]//2],
        channels=channels,reuse=reuse
    )
    output = Left_Part(middle,low_level_features,channels,num_classes,reuse=reuse)
    return tf.image.resize_bilinear(output,input_size,name='unsample')
