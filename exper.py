import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import QuantileTransformer
def pore():

    node_mean = np.load('./data/numpy_arrays/final/mean.npy')
    node_std = np.load('./data/numpy_arrays/final/std.npy')
    node_a = np.load('./data/numpy_arrays/final/a.npy')
    node_b = np.load('./data/numpy_arrays/final/b.npy')


    def _parse_function(example_proto):

        features = {
                "image_y": tf.io.FixedLenFeature((), tf.string),
                "gph_nodes": tf.io.FixedLenFeature((), tf.string),
                "gph_adj": tf.io.FixedLenFeature((), tf.string)
                #"gph_node_num" : tf.FixedLenFeature((), tf.string)
            }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        image_y = tf.io.decode_raw(parsed_features["image_y"],  tf.float32)
        gph_nodes = tf.io.decode_raw(parsed_features["gph_nodes"],  tf.float32)
        gph_adj = tf.io.decode_raw(parsed_features["gph_adj"],  tf.float32)
        #gph_node_num = tf.decode_raw(parsed_features["gph_node_num"],  tf.float32)
        
        return image_y, gph_nodes, gph_adj



    dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
    dataset = dataset.map(_parse_function)

    for i,n,a in dataset:

        dim = int(math.sqrt(int(a.shape[0])))
        i = np.reshape(i, (1,256,256,3))
        n = np.reshape(n, (dim,2))
        a = np.reshape(a, (dim,dim))
        n = (((n - node_b)/node_a)*node_std)+node_mean
        
        print(n,a)
        
        break

arr = np.load('./data/numpy_arrays/all_nodes.npy')
#print(arr[0])

wh_128 = np.where(arr == 128.0 )
print(wh_128[0])

new_arr = np.delete(arr, wh_128, 0)
print(new_arr[0:6,:])
wh_128 = np.where(new_arr == -128.0 )
new_arr = np.delete(new_arr, wh_128, 0)
test = new_arr
#print(new_arr,new_arr.shape)

plt.hist(new_arr,bins=200)
plt.show()

qt = QuantileTransformer(output_distribution='normal')

new_arr = qt.fit_transform(new_arr)

plt.hist(new_arr,bins=200)
plt.show()
#print(new)
print(arr[0:6,:])
nump = test[0:6,:]
print(nump)
nump = qt.transform(nump)
print(nump)
nump = qt.inverse_transform(nump)
print(nump)


#for i in range(len(arr)):