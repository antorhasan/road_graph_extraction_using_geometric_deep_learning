import tensorflow as tf 
#from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPool2D
import numpy as np
import math
import cv2
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, SAGEConv
#from torch.nn.functional import c

#tf.compat.v1.enable_eager_execution()

node_a = np.load('./data/numpy_arrays/range/a.npy')
node_b = np.load('./data/numpy_arrays/range/b.npy')

adj = np.load('./data/numpy_arrays/adj.npy')

def _parse_function(example_proto):

    features = {
            "image_y": tf.io.FixedLenFeature((), tf.string),
            "gph_nodes": tf.io.FixedLenFeature((), tf.string),
            "gph_adj": tf.io.FixedLenFeature((), tf.string)
        }

    parsed_features = tf.io.parse_single_example(example_proto, features)

    image_y = tf.io.decode_raw(parsed_features["image_y"],  tf.float32)
    gph_nodes = tf.io.decode_raw(parsed_features["gph_nodes"],  tf.float32)
    gph_adj = tf.io.decode_raw(parsed_features["gph_adj"],  tf.float32)
    
    return image_y, gph_nodes, gph_adj



class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.conv2 = nn.Conv2d(32,64,3, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, padding=1)
        self.conv4 = nn.Conv2d(128,256,3, padding=1)
        self.conv5 = nn.Conv2d(256,256,3, padding=1)
        
        self.sagpool = SAGPooling(256, ratio=0.8)
        self.sage = SAGEConv(205, 2)
        #self.edge_idx = 

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2, 2)
        #print(x.shape)
        x = torch.reshape(x, (256,256))
        edge = torch.Tensor(adj).long().t().contiguous().cuda()

        x , edge, _ , _,_,_ = self.sagpool(x, edge)
        x = self.sage(x.t(), edge)

        print(x.shape)
        print(edge.shape)
        print(asd)
        return x


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

    model.train()
    
    node_features = model(images)

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
#dataset = dataset.shuffle(4000)
#dataset = dataset.batch(1)

#model = model()
#model = model.sav()
#model.load_weights('./data/model/weight.h5')
#model.save('./data/model/',save_format='h5')

optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
#train_loss = tf.keras.metrics.Sum()

EPOCHS = 2
coun = 0
run_t = 0
run_nod = 0
run_adj = 0
run_num = 0
array = np.load('./data/numpy_arrays/nodes_out.npy')
qt = QuantileTransformer(output_distribution='normal')
shob = qt.fit_transform(array)

use_cuda = not False and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = Conv().to(device)

for epoch in range(EPOCHS):
    for i,n,a in dataset:
        dim = int(math.sqrt(int(a.shape[0])))
        i = np.reshape(i, (1,3,512,512))
        n = np.reshape(n, (dim,2))
        a = np.reshape(a, (dim,dim))

        i = torch.Tensor(i).cuda()
        n = torch.Tensor(n).cuda()
        a = torch.Tensor(a).cuda()

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

    model.save_weights('./data/model/weight_out_128.h5')
    #model.load_weights('./data/model/weight.h5')
    #model.save_weights('./data/model/weight_softmax.h5')


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

    #node_features = (((node_features - node_b)/node_a)*node_std)+node_mean
    node_features = (node_features - node_b)/node_a
    node_features = qt.inverse_transform(node_features)

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
    
