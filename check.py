import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPool2D
import numpy as np
import math
import cv2
tf.enable_eager_execution()
def data_check():
    

    node_mean = np.load('./data/numpy_arrays/first/mean.npy')
    node_std = np.load('./data/numpy_arrays/first/std.npy')
    node_a = np.load('./data/numpy_arrays/first/a.npy')
    node_b = np.load('./data/numpy_arrays/first/b.npy')

    num_a = np.load('./data/numpy_arrays/nodes/a.npy')
    num_b = np.load('./data/numpy_arrays/nodes/b.npy')


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

        return image_y, gph_nodes, gph_adj, gph_node_num

    dataset = tf.data.TFRecordDataset('./data/record/train_full.tfrecords')
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(2)

    for i,j,k,l in dataset:
        #l = (l-num_b)/num_a
        #l = math.ceil(l)
        #l = int(l)
        #print(k.shape[1])
        dim = int(math.sqrt(int(k.shape[0])))
        i = np.reshape(i, (256,256,3))
        k = np.reshape(k, (dim,dim))
        j = np.reshape(j, (dim,2))
        i = i*255.0
        i = np.asarray(i, dtype=np.uint8)

        j = (((j - node_b)/node_a)*node_std)+node_mean
        l = (l-num_b)/num_a
        l = tf.math.exp(l)
        print(j,k,l)
        cv2.imshow('img',i)
        cv2.waitKey(0)
        break

def reshape_check():
    arr = tf.constant([[[1,2],[3,4]],[[5,6],[7,8]]])
    print(arr)
    arr = tf.reshape(arr, [-1,2])
    print(arr)

if __name__ == "__main__":
    
    pass