import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
#tf.compat.v1.enable_eager_execution()

arr = tf.constant([[1,2,3.],[2,3,4],[3,2,1]])

#arr = np.load('./data/numpy_arrays/all_nodes.npy')
a = np.zeros([3,3])
#np.put(a, [(0,1,2,3)], [-44])
#print(a)
rows = tf.constant([0,2])
print(rows)
#rows = tf.constant(arr[0,])

columns = tf.constant([2,2])

a[rows,columns] = 1
print(tf.dtypes.cast(a,tf.float32))

#print(arr.shape)
#arr = arr[:,0]
""" for i in range(len(arr)):
    if arr[i] == 128. or arr[i] == -128. :
        arr = np.delete(arr,i) """
idx = np.where(arr == 128.)
#print(idx[1])
arr = np.delete(arr, idx[0])

idx = np.where(arr == -128.)
arr = np.delete(arr, idx[0])

plt.hist(arr,bins=200)
#plt.show()