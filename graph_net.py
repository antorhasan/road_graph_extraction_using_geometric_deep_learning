import tensorflow as tf 
from tensorflow.keras.layers import Conv2D
import numpy as np

tf.enable_eager_execution()



def _parse_function(example_proto):

    features = {
            "image_y": tf.FixedLenFeature((), tf.string),
            "gph_nodes": tf.FixedLenFeature((), tf.string),
            "gph_adj": tf.FixedLenFeature((), tf.string),
            "gph_node_num" : tf.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float32)
    gph_nodes = tf.decode_raw(parsed_features["gph_nodes"],  tf.float32)
    gph_adj = tf.decode_raw(parsed_features["gph_adj"],  tf.float32)
    gph_node_num = tf.decode_raw(parsed_features["gph_node_num"],  tf.float32)

    #size = np.asarray(gph_node_num, dtype = np.int32)
    #ize = int(size)
    #image_y = tf.reshape(image_y,[256,256,3])
    #size = tf.dtypes.cast(gph_node_num, dtype=tf.int64)
    #gph_nodes = tf.reshape(gph_nodes,[14,2])
    #gph_adj = tf.reshape(image_y,[size,size])
    #image_y = tf.reshape(image_y,[256,256,3])
    #image_y = tf.cast(image_y, dtype=tf.float32)
    gph_nodes = tf.dtypes.cast(gph_nodes,dtype = tf.float32)
    """ mean = np.load('./data/numpy_arrays/thin_line/mean.npy')
    std = np.load('./data/numpy_arrays/thin_line/std.npy')
    a = np.load('./data/numpy_arrays/thin_line/a.npy')
    b = np.load('./data/numpy_arrays/thin_line/b.npy')

    image_y = (image_y-mean)/std

    image_y = (image_y*a) + b  """ 

    return image_y, gph_nodes, gph_adj, gph_node_num



class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        
        self.conv1 = Conv2D(100,bias_initializer=tf.keras.initializers.constant(.01),activation='relu')
        self.dense1 = Dense(100, activation='relu',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')
        self.dense2 = Dense(1, activation='tanh',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.lstm(inputs)
        #x = self.dense1(x)
        x = self.dense2(x)
        return x

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def model(self):
        x = tf.keras.layers.Input(shape=(27, 1))

        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()

dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.batch(3)



for i,j,k,l in dataset:
    
    i = np.reshape(i, (256,256,3))
    k = np.reshape(k, (14,14))
    k = np.transpose(k)
    #j = np.reshape(j, (int(l),2))
    #k = np.reshape(i, (256,256,3))
    #l = np.reshape(i, (256,256,3))
    #print(l)
    print(i,j,k,l)

    break