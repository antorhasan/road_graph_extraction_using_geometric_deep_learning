import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense
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


class SLayer():
    def __init__(self,num_nodes ):
        super(SLayer, self).__init__()
        #self.num_nodes = num_nodes
        self.filters = None


    def call(self, input, num_nodes):
        self.filters = num_nodes

class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        #self.input_dim = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",shape=[int(input_shape[-1]),self.num_outputs])
        self.w = self.add_weight(shape=(),
                             initializer='random_normal',
                             trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

    def call(self, inputs):
        output = tf.nn.conv2d(inputs, W, strides=stride, padding="VALID", name="conv")
        return tf.matmul(input, self.kernel)



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

        '''num nodes ops'''
        self.conv10 = Conv2D(1,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='hard_sigmoid',kernel_initializer='he_normal')
        self.flat1 = Flatten()
        self.dense1 = Dense(1,bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal', activation='sigmoid')

        '''adj mat ops'''
        self.conv11 = SLayer()
        #self.dense1 = Dense(100, activation='relu',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')
        #self.dense2 = Dense(1, activation='tanh',bias_initializer=tf.keras.initializers.constant(.01),kernel_initializer='he_normal')

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

        a = self.conv10(o)
        a = self.flat1(a)
        a = self.dense1(a)

        num_nodes = int(208*a)
        s = self.conv11(o,num_nodes)


        return s

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


model = MyModel()
model.model()


#dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
#dataset = dataset.map(_parse_function)
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