import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax
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
    
    return image_y, gph_nodes, gph_adj, gph_node_num



class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(5,(3,3),strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv2 = Conv2D(8,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv3 = Conv2D(12,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv4 = Conv2D(16,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv5 = Conv2D(16,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv6 = Conv2D(32,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv7 = Conv2D(32,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv8 = Conv2D(32,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv9 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv10 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv11 = Conv2D(64,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv12 = Conv2D(96,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv13 = Conv2D(96,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv14 = Conv2D(128,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv15 = Conv2D(128,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv16 = Conv2D(176,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.conv17 = Conv2D(176,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        o = self.conv17(x)
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

    def __init__(self):
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

    def __init__(self):
        super(AdjLayer, self).__init__()
        self.conv = Conv2D(156,(3,3),bias_initializer=tf.keras.initializers.constant(.01),activation=None,kernel_initializer='he_normal')
        self.soft = Softmax(axis=1) #row-wise softmax

    def call(self, inputs, adj):
        s = self.conv(inputs)
        s = tf.reshape(s,[-1,156])
        s = self.soft(s)
        Sout = s
        new_weird = tf.ones([48400, 48400])
        temp = tf.linalg.matmul(s,new_weird,transpose_a=True)
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

def loss_object(node_attr_lab, adj_mat_lab, node_num_lab, node_attr_pred, adj_mat_pred, node_num_pred):


    node_loss = tf.keras.losses.mse(node_attr_lab, node_attr_pred)
    adj_loss = -0.65*tf.reduce_mean(tf.math.multiply(adj_mat_lab,tf.math.log(adj_mat_pred)))-(1-0.65)*tf.reduce_mean(tf.math.multiply((1-adj_mat_lab),tf.math.log(1-adj_mat_pred)))
    num_loss = tf.keras.losses.mse(node_num_lab, node_num_pred)

    total = node_loss+adj_loss+num_loss
    return node_loss,adj_loss,num_loss,total

class allmodel(tf.keras.Model):

    def __init__(self):
        super(allmodel, self).__init__()
        self.org_mod = MyModel()
        self.num_mod = NumLayer()
        self.adj_mod = AdjLayer()
        self.node_mod = NodeLayer()
    
    def call(self,images):
        org = self.org_mod(images)
        num_nodes, adj = self.num_mod(org)
        new_adj, Sout = self.adj_mod(org, adj)
        node_features = self.node_mod(org, Sout)
        return node_features, new_adj, num_nodes

@tf.function
def train_step(images, node_attr_lab, adj_mat_lab, node_num_lab):

    with tf.GradientTape() as tape:
        """ org = org_model(images)
        num_nodes, adj = num_model(org)
        new_adj, Sout = adj_model(org, adj)
        node_features = node_model(org, Sout) """
        node_features, new_adj, num_nodes = model(images)

        paddings_adj = tf.constant([[0, 156-node_num_lab], [0, 156-node_num_lab]])
        adj_mat_lab = tf.pad(adj_mat_lab, paddings_adj, "CONSTANT")

        paddings_node = tf.constant([[0, 156-node_num_lab], [0, 0]])
        node_attr_lab = tf.pad(node_attr_lab, paddings_node, "CONSTANT")

        node_loss, adj_loss, num_loss, total = loss_object(node_attr_lab, adj_mat_lab, node_num_lab, node_features, new_adj, num_nodes)
        #train_loss.update_state([node_loss, adj_loss, num_loss])
    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total


dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(400)
dataset = dataset.batch(1)

model = allmodel()

optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
#train_loss = tf.keras.metrics.Sum()

EPOCHS = 2
for epoch in range(EPOCHS):
    for i,j,k,l in dataset:
        l = (l-num_b)/num_a
        l = math.ceil(l)
        i = np.reshape(i, (1,256,256,3))
        k = np.reshape(k, (l,l))
        j = np.reshape(j, (l,2))
        metric = train_step(i,j,k,l)
    
    template = 'Epoch {}, Loss: {}, Train Loss: {},'
    print(template.format(epoch+1,
                        metric))

    #train_loss.reset_states()