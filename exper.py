import tensorflow as tf 
#tf.enable_eager_execution()

logits = tf.constant([[1.0,0,1.0],[0,1.0,0],[0,0,1.0]])
softmax = tf.math.exp(logits) / tf.math.reduce_sum(tf.math.exp(logits))

print(softmax)