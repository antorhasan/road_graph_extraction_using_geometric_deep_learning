import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPool2D
import numpy as np
import math
import cv2

#tf.compat.v1.enable_eager_execution()

node_mean = np.load('./data/numpy_arrays/final/mean.npy')
node_std = np.load('./data/numpy_arrays/final/std.npy')
node_a = np.load('./data/numpy_arrays/final/a.npy')
node_b = np.load('./data/numpy_arrays/final/b.npy')

#num_a = np.load('./data/numpy_arrays/nodes/a.npy')
#num_b = np.load('./data/numpy_arrays/nodes/b.npy')


def _parse_function(example_proto):

    features = {
            "image_y": tf.io.FixedLenFeature((), tf.string),
            "gph_nodes": tf.io.FixedLenFeature((), tf.string),
            "gph_adj": tf.io.FixedLenFeature((), tf.string)
            #"gph_node_num" : tf.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image_y = tf.io.decode_raw(parsed_features["image_y"],  tf.float32)
    gph_nodes = tf.io.decode_raw(parsed_features["gph_nodes"],  tf.float32)
    gph_adj = tf.io.decode_raw(parsed_features["gph_adj"],  tf.float32)
    #gph_node_num = tf.decode_raw(parsed_features["gph_node_num"],  tf.float32)
    
    return image_y, gph_nodes, gph_adj



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
        #self.conv5 = Conv2D(2,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='relu',kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.max2(x)
        x = self.conv3(x)
        x = self.max3(x)
        x = self.conv4(x)
        x = self.max4(x)
        #x = self.conv5(x)
        #print(x)
        return x



class Assign(tf.keras.layers.Layer):

    def __init__(self):
        super(Assign, self).__init__()
        self.conv = Conv2D(156,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation=None,kernel_initializer='he_normal')
        self.soft = Softmax(axis=1) #row-wise softmax
        self.convn = Conv2D(2,(3,3),padding='same',bias_initializer=tf.keras.initializers.constant(.01),activation='tanh',kernel_initializer='he_normal')

    def call(self, inputs):
        s = self.conv(inputs)
        s = tf.reshape(s,[-1,156])
        s = self.soft(s)
        

        new_nodes = self.convn(inputs)
        #print(new_nodes)
        new_nodes = tf.reshape(new_nodes,[-1,2])
        #print(new_nodes)
        new_nodes = tf.linalg.matmul(s,new_nodes,transpose_a=True)
        #new_nodes = tf.math.tanh(new_nodes)
        #print(new_nodes)

        """ 
        s_ass = np.zeros((tf.shape(s)[0],tf.shape(s)[1]))
        idx = tf.math.argmax(s,axis=1)

        for i in range(tf.shape(s)[0]):
            s_ass[i,idx[i]] = 1.

        #s_ass = np.where(s > 0.5, 1.0, 0.0)

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
            new_nodes[i,:] = mean """

        new_adj = tf.linalg.matmul(s,s,transpose_a=True)
        new_adj = tf.reshape(new_adj,[156,156])
        #new_adj = tf.where(new_adj > 0.5, 1.0, 0)
        new_nodes = tf.dtypes.cast(new_nodes, dtype=tf.float32)

        return new_nodes, new_adj

class model(tf.keras.Model):

    def __init__(self):
        super(model, self).__init__()
        self.conv = Conv()
        self.assign = Assign()
    
    def call(self,images):
        x = self.conv(images)
        n, a = self.assign(x)
        #print(n)
        return n, a


def loss_object(node_attr_lab, adj_mat_lab, node_attr_pred, adj_mat_pred):

    node_loss = tf.reduce_mean(tf.keras.losses.mse(node_attr_lab, node_attr_pred))
    #adj_loss = -0.6*tf.reduce_mean(tf.math.multiply(adj_mat_lab,tf.math.log(adj_mat_pred)))-(1-0.6)*tf.reduce_mean(tf.math.multiply((1-adj_mat_lab),tf.math.log(1-adj_mat_pred)))
    #num_loss = 4*tf.reduce_mean(tf.keras.losses.mse(node_num_lab, node_num_pred))

    """ #print(node_attr_pred)
    node_attr_pred = (((node_attr_pred - node_b)/node_a)*node_std)+node_mean
    node_attr_lab = (((node_attr_lab - node_b)/node_a)*node_std)+node_mean
    #print(node_attr_pred)
    node_attr_pred = tf.where(node_attr_pred > 128, 128.0, node_attr_pred)
    node_attr_pred = tf.where(node_attr_pred < -128, -128.0, node_attr_pred)
    #print(node_attr_pred)

    node_p = np.zeros((256,256))
    rows_p = tf.dtypes.cast(node_attr_pred[:,1],dtype=tf.int32)
    columns_p = tf.dtypes.cast(node_attr_pred[:,0],dtype=tf.int32)
    node_p[rows_p,columns_p] = 1.0
    node_attr_pred = tf.dtypes.cast(node_p, tf.float32)
    node_attr_pred = tf.nn.softmax(node_attr_pred)

    node_l = np.zeros((256,256))
    rows_l = tf.dtypes.cast(node_attr_lab[:,1],dtype=tf.int32)
    columns_l = tf.dtypes.cast(node_attr_lab[:,0],dtype=tf.int32)
    node_l[rows_l,columns_l] = 1.0
    node_attr_lab = tf.dtypes.cast(node_l, tf.float32)
    node_attr_lab = tf.nn.softmax(node_attr_lab) """

    #node_loss = tf.compat.v1.losses.absolute_difference(node_attr_lab, node_attr_pred)
    adj_loss = tf.compat.v1.losses.absolute_difference(adj_mat_lab, adj_mat_pred)
    total = node_loss+adj_loss
    return node_loss,adj_loss,total

#@tf.function
def train_step(images, node_attr_lab, adj_mat_lab, dim):

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


        node_loss, adj_loss, total = loss_object(node_attr_lab, adj_mat_lab, node_features, new_adj)
        #train_loss.update_state([node_loss, adj_loss, num_loss])
    gradients = tape.gradient(total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    #return total
    return total,node_loss, adj_loss
    
dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(4000)
#dataset = dataset.batch(1)

model = model()
#model = model.sav()
#model.load_weights('./data/model/weight.h5')
#model.save('./data/model/',save_format='h5')

optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
#train_loss = tf.keras.metrics.Sum()

EPOCHS = 5
coun = 0
run_t = 0
run_nod = 0
run_adj = 0
run_num = 0
for epoch in range(EPOCHS):
    for i,n,a in dataset:
        #l = (l-num_b)/num_a
        #l = math.ceil(l)
        #l = int(l)
        #print(k.shape[1])
        
        dim = int(math.sqrt(int(a.shape[0])))
        i = np.reshape(i, (1,256,256,3))
        n = np.reshape(n, (dim,2))
        a = np.reshape(a, (dim,dim))
        #node_features = (((n - node_b)/node_a)*node_std)+node_mean
        #print(i,node_features,a)
        #print(i,j,k,l)
        #metric = train_step(i,j,k,l,dim)
        metric, node_loss, adj_loss = train_step(i,n,a,dim)

        run_t = run_t + metric/2000
        run_nod = run_nod + node_loss/2000
        run_adj = run_adj + adj_loss/2000
        #run_num = run_num + num_loss/2000
        coun+=1
        if coun%2000==0 :
            template = 'Epoch {}, Loss: {}, nod_Loss: {}, adj_Loss: {} '
            #print(template.format(epoch+1,metric, nod e_loss, adj_loss, num_loss))
            print(template.format(epoch+1,run_t, run_nod, run_adj))
            run_t = 0
            run_nod = 0
            run_adj = 0
            #run_num = 0
            #`break
#model.load_weights('./data/model/weight.h5')
model.save_weights('./data/model/weight_softmax.h5')


dataset_test = tf.data.TFRecordDataset('./data/record/val.tfrecords')
dataset_test = dataset_test.map(_parse_function)
#dataset_test = dataset_test.shuffle(6000)

counter = 0
for i,n,a in dataset:
    dim = int(math.sqrt(int(a.shape[0])))
    i = np.reshape(i, (1,256,256,3))
    node_features, new_adj = model(i)

    #pred_dim = (num_nodes-num_b)/num_a
    #pred_dim = tf.math.exp(pred_dim)
    #pred_dim = np.array(pred_dim)
    #pred_dim = pred_dim.item()
    #pred_dim = int(pred_dim)


    #new_adj = new_adj[0:pred_dim,0:pred_dim]
    #node_features = node_features[0:pred_dim,:]
    #new_adj = new_adj[0:pred_dim,0:pred_dim]
    #node_features = node_features[0:pred_dim,:]

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
    
