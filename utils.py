import tensorflow as tf


weight_init = tf.contrib.layers.variance_scaling_initializer()
weight_regularizer = tf.contrib.layers.l2_regularizer(0.0001)

def batch_norm(x,is_training=True,scope='batch_norm'):
    layer = tf.contrib.layers.batch_norm(x,is_training=is_training,scope=scope)
    return layer

def conv(x,channels,kernel=4,strides=2,padding='SAME',use_bias=True,scope='conv_0',dilation_rate=1):
    with tf.variable_scope(scope):
        layer = tf.layers.conv2d(
            inputs=x,
            filters=channels,
            kernel_size=kernel,
            kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer,
            strides=strides,
            use_bias=use_bias,
            padding=padding,
            dilation_rate=dilation_rate
        )
    return layer

def bottle_resblock(x,channels,is_training=True,use_bias=True,downsample=False,scope='bottle_resblock'):
    with tf.variable_scope(scope):
        x = batch_norm(x,is_training,scope='batch_norm_0')
        shortcut = tf.nn.relu(x)
        
        x = conv(shortcut,channels,kernel=1,strides=1,use_bias=use_bias,scope='conv_1x1_front')
        x = batch_norm(x,is_training,scope='batch_norm_3x3')
        x = tf.nn.relu(x)
        
        if downsample == True:
            x = conv(x,channels,kernel=3,strides=2,use_bias=use_bias,scope='conv_0')
            shortcut = conv(shortcut,channels*4,kernel=1,strides=2,use_bias=use_bias,scope='conv_init')
        else:
            x = conv(x,channels,kernel=3,strides=1,use_bias=use_bias,scope='conv_0')
            shortcut = conv(shortcut,channels*4,kernel=1,strides=1,use_bias=use_bias,scope='conv_init')
        
        x = batch_norm(x,is_training,scope='batch_norm_1x1_back')
        x = tf.nn.relu(x)
        x = conv(x,channels*4,kernel=1,strides=1,use_bias=use_bias,scope='conv_1x1_back')
        
        return x + shortcut
    
def fully_connected(x,units,use_bias=True,scope='fully_0'):
    with tf.variable_scope(scope):
        # x = tf.layers.flatten(x)
        x = tf.layers.dense(
            x,units=units,kernel_initializer=weight_init,
            kernel_regularizer=weight_regularizer,use_bias=use_bias,
        )
        return x
    
def layer_norm(x,scope='layer_norm_0'):
    with tf.variable_scope(scope):
        x = tf.contrib.layers.layer_norm(x)
        return x
    
def stem(x,dim,scope='stem'):
    with tf.variable_scope(scope):
        x = conv(x,channels=dim,kernel=2,strides=2,padding='VALID')
        x = layer_norm(x)
    return x

def downsample_layer(x,dim,scope='downsample_layer'):
    with tf.variable_scope(scope):
        x = layer_norm(x)
        x = conv(x,channels=dim,kernel=2,strides=2,padding='VALID')
    return x

def depthwise_conv2d(x,dim,kernel=7,strides=1,pad=3,scope='depthwise_conv2d_0'):
    strides = [strides]*4
    pad = [[0,0],[pad,pad],[pad,pad],[0,0]]
    with tf.variable_scope(scope):
        filter_ = tf.random.normal(shape=(kernel,kernel,dim,1))
        filter_ = tf.Variable(filter_)
        x = tf.nn.depthwise_conv2d(input=x,filter=filter_,strides=strides,padding='VALID')
        x = tf.pad(x,pad)
    return x

def gelu(x,scope='gelu_0'):
    with tf.variable_scope(scope):
        x = 0.5*x*(1.0 + tf.math.tanh(0.7978845608028654*(x+0.044715*tf.math.pow(x,3))))
    return x

def drop_path(x,drop_prob=0.2,is_training=True):
    if drop_prob == 0 or is_training == False: return x
    keep_prob = 1 - drop_prob
    shape = (tf.shape(x)[0],) + (1,)*(len(x.shape)-1)
    random_tensor = keep_prob + tf.random_uniform(shape=shape,minval=0,maxval=1)
    random_tensor = tf.floor(random_tensor)
    for i in range(len(x.shape)-1,1):
        random_tensor = tf.expand_dims(random_tensor,axis=i,dim=tf.shape(x)[i])
    output = tf.divide(x,keep_prob)*random_tensor
    return output

def block(x, dim, drop_prob=0.2, layer_scale_init_value=1e-6, is_training=True, scope='inverted_bottleneck_block'):
    with tf.variable_scope(scope):
        gamma = tf.Variable(1e-6,trainable=True)
        input_ = x
        x = depthwise_conv2d(x,dim,kernel=7,strides=1,pad=3)
        x = layer_norm(x)
        x = fully_connected(x,dim*4)
        x = gelu(x)
        x = fully_connected(x,dim,scope='fully_1')
        x = x*gamma
        x = input_ + drop_path(x,drop_prob=drop_prob,is_training=is_training)
    return x