import tensorflow as tf 

tf.enable_eager_execution()

arr = tf.constant([[[[1.,9.],[2,8],[3,7]],[[4,6],[5,5],[6,4]],[[7,3],[8,2],[9,1]]]])

print(arr[:,0:2,0:2,:])