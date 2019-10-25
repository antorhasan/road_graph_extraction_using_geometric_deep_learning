import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPool2D
import numpy as np
import math
import cv2

#tf.enable_eager_execution()

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



class Conv(tf.keras.layers.Layer):

    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = Conv2D(32,(3,3),padding='same',strides=(1,1),bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max1 = MaxPool2D((2,2))
        self.conv2 = Conv2D(64,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max2 = MaxPool2D((2,2))
        self.conv3 = Conv2D(128,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max3 = MaxPool2D((2,2))
        '''the number of filters in following convolution can be changed to increase features per node when feature maps are converted to nodes'''
        self.conv4 = Conv2D(156,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')
        self.max4 = MaxPool2D((2,2))
        self.conv5 = Conv2D(2,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.max4(x)
        x = self.conv5(x)
        
        return x


    def model(self):
        x = tf.keras.layers.Input(shape=(256,256, 3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()


class Assign(tf.keras.layers.Layer):

    def __init__(self):
        super(Assign, self).__init__()
        self.conv = Conv2D(156,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation=None,kernel_initializer='he_normal')
        self.soft = Softmax(axis=1) #row-wise softmax

    def call(self, inputs):
        s = self.conv(inputs)
        s = tf.reshape(s,[-1,156])
        s = self.soft(s)
        s_ass = tf.where(s > 0.5, 1.0, 0)
        inp = tf.reshape(inputs, [-1,2])

        new_nodes = np.zeros((tf.shape(s_ass)[1],tf.shape(inp)[1]))
        for i in range(tf.shape(s_ass)[1]):
            coor = np.where(s_ass[:,i] == 1.0)
            coor = list(coor)
            lis = []
            for j in range(len(coor[0])):
                lis.append(inp[coor[0][j],:])
            arr = np.asarray(lis)
            mean = np.mean(arr,axis=0)
            new_nodes[i,:] = mean

        new_adj = tf.linalg.matmul(s,s,transpose_a=True)
        new_adj = tf.reshape(new_adj,[156,156])
        #new_adj = tf.where(new_adj > 0.5, 1.0, 0)

        return new_nodes, new_adj

class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__()
        self.conv = Conv()
        self.assign = Assign()
    
    def call(self,images):
        x = self.conv(images)
        n, a = self.assign(x)
        return node_features, new_adj, num_nodes


def loss_object(node_attr_lab, adj_mat_lab, node_num_lab, node_attr_pred, adj_mat_pred, node_num_pred):

    node_loss = tf.reduce_mean(tf.keras.losses.mse(node_attr_lab, node_attr_pred))
    adj_loss = -0.6*tf.reduce_mean(tf.math.multiply(adj_mat_lab,tf.math.log(adj_mat_pred)))-(1-0.6)*tf.reduce_mean(tf.math.multiply((1-adj_mat_lab),tf.math.log(1-adj_mat_pred)))
    num_loss = 4*tf.reduce_mean(tf.keras.losses.mse(node_num_lab, node_num_pred))

    total = node_loss+adj_loss+num_loss
    return node_loss,adj_loss,num_loss,total

#@tf.function
def train_step(images, node_attr_lab, adj_mat_lab, node_num_lab, dim):

    with tf.GradientTape() as tape:
        """ org = org_model(images)
        num_nodes, adj = num_model(org)
        new_adj, Sout = adj_model(org, adj)
        node_features = node_model(org, Sout) """
        node_features, new_adj = model(images)

        #num_nodes = 156

        #pred_dim = (num_nodes-num_b)/num_a
        #pred_dim = tf.math.exp(pred_dim)
        #pred_dim = np.array(pred_dim)
        #print(pred_dim.shape)
        #pred_dim = pred_dim.item()
        #pred_dim = list(pred_dim)
        #pred_dim = pred_dim.numpy()
        #pred_dim = np.asscalar(pred_dim)
        #pred_dim = pred_dim.item()
        #pred_dim = pred_dim.tolist()
        pred_dim = 156
        
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
            node_features = node_features[0:pred_dim,:]


        node_loss, adj_loss, num_loss, total = loss_object(node_attr_lab, adj_mat_lab, node_features, new_adj)
        #train_loss.update_state([node_loss, adj_loss, num_loss])
    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #return total
    return total,node_loss, adj_loss, num_loss
    
dataset = tf.data.TFRecordDataset('./data/record/train_25.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(6000)
#dataset = dataset.batch(1)

model = allmodel()
#model = model.sav()
#model.load_weights('./data/model/weight.h5')
#model.save('./data/model/',save_format='h5')

optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)
#train_loss = tf.keras.metrics.Sum()

EPOCHS = 2
coun = 0
run_t = 0
run_nod = 0
run_adj = 0
run_num = 0
for epoch in range(EPOCHS):
    for i,j,k,l in dataset:
        #l = (l-num_b)/num_a
        #l = math.ceil(l)
        #l = int(l)
        #print(k.shape[1])
        dim = int(math.sqrt(int(k.shape[0])))
        i = np.reshape(i, (1,256,256,3))
        k = np.reshape(k, (dim,dim))
        j = np.reshape(j, (dim,2))
        #print(i,j,k,l)
        #metric = train_step(i,j,k,l,dim)
        metric, node_loss, adj_loss, num_loss = train_step(i,j,k,l,dim)

        run_t = run_t + metric/2000
        run_nod = run_nod + node_loss/2000
        run_adj = run_adj + adj_loss/2000
        run_num = run_num + num_loss/2000
        coun+=1
        if coun%2000==0 :
            template = 'Epoch {}, Loss: {}, nod_Loss: {}, adj_Loss: {}, num_Loss: {}, '
            #print(template.format(epoch+1,metric, nod e_loss, adj_loss, num_loss))
            print(template.format(epoch+1,run_t, run_nod, run_adj, run_num))
            run_t = 0
            run_nod = 0
            run_adj = 0
            run_num = 0
            #break
#model.load_weights('./data/model/weight.h5')
#model.save_weights('./data/model/weight.h5')


dataset_test = tf.data.TFRecordDataset('./data/record/train_15.tfrecords')
dataset_test = dataset_test.map(_parse_function)
dataset_test = dataset_test.shuffle(6000)

counter = 0
for i,j,k,l in dataset:
    dim = int(math.sqrt(int(k.shape[0])))
    i = np.reshape(i, (1,256,256,3))
    node_features, new_adj, num_nodes = model(i)

    pred_dim = (num_nodes-num_b)/num_a
    pred_dim = tf.math.exp(pred_dim)
    pred_dim = np.array(pred_dim)
    pred_dim = pred_dim.item()
    pred_dim = int(pred_dim)

    new_adj = new_adj[0:pred_dim,0:pred_dim]
    node_features = node_features[0:pred_dim,:]
    node_features = (((node_features - node_b)/node_a)*node_std)+node_mean
    new_adj = np.where(new_adj>.5, 1.0 , 0)
    np.savetxt('./data/output/adj'+str(counter)+'.txt', new_adj)
    np.savetxt('./data/output/node'+str(counter)+'.txt',node_features)
    image = i*255.0
    image = np.reshape(image,(256,256,3))
    image = np.asarray(image, dtype=np.uint8)

    cv2.imwrite('./data/output/img'+str(counter)+'.png',image)
    counter+=1
    if counter==6:
        break
    
