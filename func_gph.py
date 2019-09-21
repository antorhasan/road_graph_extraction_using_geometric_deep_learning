import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPool2D
import numpy as np
import math

tf.enable_eager_execution()

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
    
    dim = tf.dtypes.cast(tf.shape(gph_adj), dtype=tf.float32) 
    dim = tf.math.sqrt(dim)
    dim = tf.dtypes.cast(dim, dtype=tf.int32)
    #pr = tf.print(dim)
    print(dim)
    dim_adj = tf.fill([1,2], dim[0])
    image_y = tf.reshape(image_y, [256,256,3])
    gph_adj = tf.reshape(gph_adj, dim_adj)
    gph_nodes = tf.reshape(gph_nodes, [dim[0],2])

    return image_y, [gph_nodes, gph_adj, gph_node_num]



class MyModel(tf.keras.layers.Layer):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32,(3,3),padding='same',strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv2 = Conv2D(32,(3,3),padding='same',strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max1 = MaxPool2D((2,2))
        self.conv3 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv4 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max2 = MaxPool2D((2,2))
        self.conv5 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv6 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max3 = MaxPool2D((2,2))
        self.conv7 = Conv2D(256,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv8 = Conv2D(256,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        
        """ self.conv9 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv10 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv11 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv12 = Conv2D(96,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv13 = Conv2D(96,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv14 = Conv2D(128,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv15 = Conv2D(128,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv16 = Conv2D(176,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv17 = Conv2D(176,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
 """

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.max1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.max3(x)
        x = self.conv7(x)
        o = self.conv8(x)

        """ x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        o = self.conv17(x) """
        
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

#model = MyModel()
#model.model()

class NumLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(NumLayer, self).__init__()
        self.conv10 = Conv2D(1,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
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


class AdjLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(AdjLayer, self).__init__()
        self.conv = Conv2D(156,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation=None,kernel_initializer='he_normal')
        self.soft = Softmax(axis=1) #row-wise softmax
        self.conv1 = Conv2D(1,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='sigmoid',kernel_initializer='he_normal')

    def call(self, inputs, adj):
        s = self.conv(inputs)
        s = tf.reshape(s,[-1,156])
        s = self.soft(s)
        Sout = s
        """ new_weird = tf.ones([900, 900])
        temp = tf.linalg.matmul(s,new_weird,transpose_a=True)
        new_adj = tf.linalg.matmul(temp,s) """
        new_adj = tf.linalg.matmul(s,s,transpose_a=True)
        new_adj = tf.reshape(new_adj,[1,156,156,1])              #trying conv + sigmoid
        new_adj = self.conv1(new_adj)
        new_adj = tf.reshape(new_adj,[156,156])
        #new_adj = tf.math.sigmoid(new_adj)               #trying only sigmoid transformation
        return new_adj, Sout


class NodeLayer(tf.keras.layers.Layer):

    def __init__(self):
        super(NodeLayer,self).__init__()
        self.conv = Conv2D(2,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='tanh',kernel_initializer='he_normal')

    def call(self,inputs,Sout):
        n = self.conv(inputs)
        n = tf.reshape(n,[-1,2])
        node_features = tf.linalg.matmul(Sout,n,transpose_a=True)
        #node_features = tf.math.tanh(node_features)    #trying to keep range same according to label
        return node_features

def f_loss():
    def cost(labels, logits):

        """ pred_dim = (logits[2]-num_b)/num_a
        pred_dim = tf.math.exp(pred_dim)

        #dim = labels[]

        if pred_dim > dim :

            paddings_adj = tf.constant([[0, pred_dim - dim], [0, pred_dim - dim]])
            adj_mat_lab = tf.pad(adj_mat_lab, paddings_adj, "CONSTANT")

            paddings_node = tf.constant([[0, pred_dim - dim], [0, 0]])
            node_attr_lab = tf.pad(node_attr_lab, paddings_node, "CONSTANT")

            new_adj = new_adj[0:pred_dim,0:pred_dim]
            node_features = node_features[0:pred_dim,:]

        

        elif pred_dim < dim :
            new_adj = new_adj[0:dim,0:dim]
            node_features = node_features[0:dim,:]

        elif pred_dim == dim :
            new_adj = new_adj[0:pred_dim,0:pred_dim]
            node_features = node_features[0:pred_dim,:] """


        node_loss = tf.reduce_mean(tf.keras.losses.mse(labels[0], logits[0]))
        adj_loss = -0.65*tf.reduce_mean(tf.math.multiply(labels[1],tf.math.log(logits[1])))-(1-0.65)*tf.reduce_mean(tf.math.multiply((1-labels[1]),tf.math.log(1-logits[1])))
        num_loss = 4*tf.reduce_mean(tf.keras.losses.mse(labels[2], logits[2]))

        total = node_loss+adj_loss+num_loss

        return total
    return cost

""" def loss_object(node_attr_lab, adj_mat_lab, node_num_lab, node_attr_pred, adj_mat_pred, node_num_pred):

    node_loss = tf.reduce_mean(tf.keras.losses.mse(node_attr_lab, node_attr_pred))
    adj_loss = -0.65*tf.reduce_mean(tf.math.multiply(adj_mat_lab,tf.math.log(adj_mat_pred)))-(1-0.65)*tf.reduce_mean(tf.math.multiply((1-adj_mat_lab),tf.math.log(1-adj_mat_pred)))
    num_loss = 4*tf.reduce_mean(tf.keras.losses.mse(node_num_lab, node_num_pred))

    total = node_loss+adj_loss+num_loss
    return node_loss,adj_loss,num_loss,total """

dataset = tf.data.TFRecordDataset('./data/record/train_full.tfrecords')
dataset = dataset.map(_parse_function)
#dataset = dataset.shuffle(10000)
dataset = dataset.batch(1)

inputs = tf.keras.Input(shape=(256,256,3))

org = MyModel()(inputs)
num_nodes, adj = NumLayer()(org)
new_adj, Sout = AdjLayer()(org, adj)
node_features = NodeLayer()(org, Sout)

predictions = [node_features, new_adj, num_nodes]

model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=.00001),
              loss=f_loss())

model.fit(dataset , batch_size=1, epochs=2)
