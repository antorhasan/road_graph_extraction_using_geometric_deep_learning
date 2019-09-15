import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Softmax

tf.enable_eager_execution()

arr = tf.constant([[[[1.,9.],[2,8],[3,7]],[[4,6],[5,5],[6,4]],[[7,3],[8,2],[9,1]]]])

class model(tf.keras.Model):

    def __init__(self):
        super(model,self).__init__()
        self.flat = Flatten()
        self.soft = Softmax(axis=1)
    
    def call(self,input):
        #x = self.flat(input)
        x = tf.reshape(input, [-1,2])
        #x = tf.nn.softmax(x)
        x = self.soft(x)
        #x = tf.math.add(x[:,0],x[:,1])
        return x

model1 = model()
output = model1(arr)

#new = Flatten()
#fine = new(arr)
print(output)