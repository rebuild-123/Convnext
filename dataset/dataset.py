import tensorflow as tf

def norm_image(image):
    image_decoded = image/255.0
    return image_decoded

def load_image_label(filename, label):    
    image_string = tf.read_file(filename)      
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.cast(image_decoded,dtype=tf.float32)
    image_decoded = norm_image(image_decoded)

    label_string = tf.read_file(label)     
    label_decoded = tf.image.decode_image(label_string)
    label_decoded = tf.cast(label_decoded,dtype=tf.uint8)
    
    return image_decoded, label_decoded

def Dataset(imgs_splits_path,labs_splits_path,batchsize):
    
    with open(imgs_splits_path,'r') as f:
        filenames = sorted(f.read().splitlines())
    with open(labs_splits_path,'r') as f:
        labels = sorted(f.read().splitlines())
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_image_label)   
    dataset = dataset.batch(batchsize) 

    return dataset,len(filenames)

def load_image_label_2(filename, label):    
    image_string = tf.read_file(filename)      
    image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.cast(image_decoded,dtype=tf.float32)
    image_decoded = norm_image(image_decoded)

    label_string = tf.read_file(label)     
    label_decoded = tf.image.decode_image(label_string)
    label_decoded = tf.cast(label_decoded,dtype=tf.uint8)
    
    return image_decoded, label_decoded, filename

def Dataset_2(imgs_splits_path,labs_splits_path,batchsize):
    
    with open(imgs_splits_path,'r') as f:
        filenames = sorted(f.read().splitlines())
    with open(labs_splits_path,'r') as f:
        labels = sorted(f.read().splitlines())
    
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(load_image_label_2)   
    dataset = dataset.batch(batchsize) 

    return dataset,len(filenames)

def Test_Dataset(imgs_splits_path,batchsize):
    
    def _parseone(filename, label):    
        """ Reading and handle  image"""
        image_string = tf.read_file(filename)      
        image_decoded = tf.image.decode_image(image_string)
        image_decoded = tf.cast(image_decoded,dtype=tf.float32)
        image_decoded = _norm_image(image_decoded)
        return image_decoded

    with open(imgs_splits_path,'r') as f:
        filenames = sorted(f.read().splitlines())
    
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(_parseone)   
    dataset = dataset.batch(batchsize) 

    return dataset,len(filenames)