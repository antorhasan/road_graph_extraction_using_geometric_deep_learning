import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax
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
        self.conv1 = Conv2D(8,(9,9),strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv2 = Conv2D(16,(9,9),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv3 = Conv2D(32,(7,7),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv4 = Conv2D(64,(7,7),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv5 = Conv2D(80,(7,7),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv6 = Conv2D(96,(5,5),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv7 = Conv2D(128,(5,5),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv8 = Conv2D(144,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv9 = Conv2D(176,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        o = self.conv9(x)
        return o

    def compute_output_shape(self, input_shape):
        # You need to override this function if you want to use the subclassed model
        # as part of a functional-style model.
        # Otherwise, this method is optional.
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

    def model(self):
        x = tf.keras.layers.Input(shape=(256,256, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


class NumLayer(tf.keras.Model):

    def __init__(self,num_nodes):
        super(NumLayer, self).__init__()
        self.conv10 = Conv2D(1,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='hard_sigmoid',kernel_initializer='he_normal')
        self.flat1 = Flatten()
        self.dense1 = Dense(1,bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal', activation='tanh')

    def call(self, inputs):
        e = self.conv10(inputs)
        a = self.flat1(e)
        a = self.dense1(a)
        num_nodes = a
        adj = e
        #num_nodes = int(208*a)
        return num_nodes, adj


class AdjLayer(tf.keras.Model):

    def __init__(self,num_nodes):
        super(AdjLayer, self).__init__()
        self.conv = Conv2D(num_nodes,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation=None,kernel_initializer='he_normal')
        self.soft = Softmax(axis=1) #row-wise softmax

    def call(self, inputs, num_nodes, adj):
        s = self.conv(inputs)
        s = tf.reshape(s,[-1,num_nodes])
        s = self.soft(s)
        Sout = s
        temp = tf.linalg.matmul(s,adj,transpose_a=True)
        new_adj = tf.linalg.matmul(temp,s)
        return new_adj, Sout


class NodeLayer(tf.keras.Model):

    def __init__(self):
        super(NodeLayer,self).__init__()
        self.conv = Conv2D(2,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self,inputs,Sout):
        n = self.conv(inputs)
        n = tf.reshape(n,[-1,2])
        node_features = tf.linalg.matmul(Sout,n,transpose_a=True)
        return node_features



#model = MyModel()
#model.model()



dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.batch(3)



for i,j,k,l in dataset:
    
    i = np.reshape(i, (256,256,3))
    k = np.reshape(k, (9,9))
    #k = np.transpose(k)
    j = np.reshape(j, (9,2))
    #k = np.reshape(i, (256,256,3))
    #l = np.reshape(i, (256,256,3))
    #print(l)
    print(i,j,k,l)
    #print(l)
    break