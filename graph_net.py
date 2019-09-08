import tensorflow as tf 
from tensorflow.keras.layers import Conv2D




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