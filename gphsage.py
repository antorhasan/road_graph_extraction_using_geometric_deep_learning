import tensorflow as tf 
import networkx as nx 
import numpy as np
import math
import cv2
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, SAGEConv
from torch_geometric.nn import dense_diff_pool
import torch.optim as optim
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import convert,to_dense_adj, negative_sampling
from torch_geometric.data import Data
#from torch.nn.functional import c

#tf.compat.v1.enable_eager_execution()

node_a = np.load('./data/numpy_arrays/range/a.npy')
node_b = np.load('./data/numpy_arrays/range/b.npy')

ori_adjacen = np.load('./data/numpy_arrays/adj.npy')

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


class Bonv(nn.Module):

    def __init__(self):
        super(Bonv, self).__init__()

        self.sage1 = SAGEConv(2,2, bias= True)
        self.sage2 = SAGEConv(2,128, bias= True)
        self.sage3 = SAGEConv(2,1, bias= True)
        #self.fc1 = nn.Linear(256, 128)

    def forward(self, nodes, adjs):
        edge, _ = dense_to_sparse(adjs)
        x = self.sage1(nodes, edge)
        s = self.sage2(nodes, edge)
        x, edge, link_loss, ent_loss = dense_diff_pool(x, adjs, s)

        x_out = torch.reshape(x, (128,2))
        edge = torch.reshape(edge, (128,128))
        for i in range(edge.size(0)):
            edge[i,:] = torch.where(edge[i,:] == torch.max(edge[i,:]),torch.ones(1,128).cuda(), torch.zeros(1,128).cuda())

        edge, _ = dense_to_sparse(edge)
        x = self.sage3(x_out, edge)
        



        x = torch.reshape(x, (128,))

        return x, edge, link_loss, ent_loss,x_out


class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding=1)
        self.conv2 = nn.Conv2d(32,64,3, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, padding=1)
        self.conv4 = nn.Conv2d(128,256,3, padding=1)
        self.conv5 = nn.Conv2d(256,256,3, padding=1)
        
        #self.sagpool = SAGPooling(256, ratio=0.8,min_score=None)
        self.sage1 = SAGEConv(256, 100, bias =True, normalize=False)
        self.sage2 = SAGEConv(256, 128, bias =True, normalize=False)
        self.sage3 = SAGEConv(100, 2, bias =True, normalize=False)

        self.sage4 = SAGEConv(128, 128, bias =True, normalize=False)
        self.sage5 = SAGEConv(1, 1, bias =True, normalize=False)
        #self.fc1 = nn.Linear(128, 128)
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
        #print(x)
        org = torch.reshape(x, (256,256))

        edge = torch.Tensor(ori_adjacen).long().t().contiguous().cuda()

        x = self.sage1(org, edge)
        s = self.sage2(org, edge)
        x = torch.reshape(x, (1,256,2))

        sparse_mat = torch.Tensor(convert.to_scipy_sparse_matrix(edge).todense()).cuda()
        sparse_mat = torch.reshape(sparse_mat, (1,256,256))

        x, edge, link_loss, ent_loss = dense_diff_pool(x, sparse_mat, s)
        x = torch.tanh(x)
        
        x = torch.reshape(x, (128,2))
        
        edge = torch.reshape(edge, (128,128))
        for i in range(edge.size(0)):
            edge[i,:] = torch.where(edge[i,:] == torch.max(edge[i,:]),torch.ones(1,128).cuda(), torch.zeros(1,128).cuda())
        
        edge_out = edge
        edge, _ = dense_to_sparse(edge)
        nodes_out = x
        x = self.sage3(x, edge)
        


        x = x.reshape(-1)
        #print(x.shape)
        #x = self.fc1(x)

        #print(x.shape,nodes_out.shape)
        #print(edge_out)
        #print(asd)
        
        return x, link_loss, ent_loss, nodes_out, edge_out, edge

def embd_loss(edge_idx, z, x):
    EPS = 1e-8 
    row, col = edge_idx
    loss_pos = - torch.log((z[row] * z[col]).sum(dim=-1).sigmoid() + EPS).mean()
    col_neg = torch.randint(x.size(0), (row.size(0), ), dtype=torch.long, device=row.device)
    loss_neg = - torch.log((-(z[row] * z[col_neg])).sum(dim=-1).sigmoid() + EPS).mean()
    loss = loss_pos + loss_neg
    return loss

#@tf.function
def train_step(images, node_attr_lab, adj_mat_lab, dim, optimizer,boptim):

    model.train()
    optimizer.zero_grad()
    pred_node_embd, lnk_loss1, ent_loss1, nodes_out, edges_out, edges_dense = model(images)

    bmodel.train()
    boptim.zero_grad()
    gt_node_embd, edge_gt, lnk_loss2, ent_loss2, x_out = bmodel(node_attr_lab, adj_mat_lab)

    emb_loss = embd_loss(edge_gt, gt_node_embd, x_out)
    emb_loss1 = embd_loss(edges_dense, pred_node_embd, nodes_out)

    loss = F.l1_loss(pred_node_embd, gt_node_embd) + lnk_loss1 + lnk_loss2 + ent_loss1 + ent_loss2 + emb_loss + emb_loss1
    loss.backward()
    optimizer.step()
    boptim.step()

    return loss, nodes_out, edges_out
    
dataset = tf.data.TFRecordDataset('./data/record/train.tfrecords')
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(1000)
#dataset = dataset.batch(1)Metropolis

#model = model()
#model = model.sav()
#model.load_weights('./data/model/weight.h5')
#model.save('./data/model/',save_format='h5')



EPOCHS = 5
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
optimizer = optim.Adam(model.parameters(), lr=.0001)

bmodel = Bonv().to(device)
boptim = optim.Adam(bmodel.parameters(), lr=.0001)
""" 
checkpoint = torch.load('./data/model/torch1.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
bmodel.load_state_dict(checkpoint['bmodel_state_dict'])
boptim.load_state_dict(checkpoint['boptim_state_dict']) """

for epoch in range(EPOCHS):
    for i,n,a in dataset:
        dim = int(math.sqrt(int(a.shape[0])))
        i = np.reshape(i, (1,3,512,512))
        n = np.reshape(n, (dim,2))
        a = np.reshape(a, (dim,dim))

        i = torch.Tensor(i).cuda()
        n = torch.Tensor(n).cuda()
        a = torch.Tensor(a).cuda()

        metric, nodes_out, edges_out = train_step(i,n,a,dim, optimizer, boptim)

        run_t = run_t + metric/2000
        #run_nod = run_nod + node_loss/2000
        #run_adj = run_adj + adj_loss/2000
        #run_num = run_num + num_loss/2000
        coun+=1
        if coun%2000==0 :
            template = 'Epoch {}, Loss: {} '
            #print(template.format(epoch+1,metric, nod e_loss, adj_loss, num_loss))
            print(template.format(epoch+1,run_t))
            run_t = 0
            #run_nod = 0
            #run_adj = 0
            #run_num = 0
            #`break

    #model.save_weights('./data/model/weight_out_128.h5')
    #model.load_weights('./data/model/weight.h5')
    #model.save_weights('./data/model/weight_softmax.h5')
    torch.save({
            'model_state_dict': model.state_dict(),
            'bmodel_state_dict': bmodel.state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'boptim_state_dict': boptim.state_dict(),
            }, './data/model/torch1.pt')

counter = 0
for i,n,a in dataset:
    print(counter)
    dim = int(math.sqrt(int(a.shape[0])))
    i = np.reshape(i, (1,3,512,512))
    n = np.reshape(n, (dim,2))
    a = np.reshape(a, (dim,dim))

    i = torch.Tensor(i).cuda()
    n = torch.Tensor(n).cuda()
    a = torch.Tensor(a).cuda()

    metric, nodes_out, edges_out = train_step(i,n,a,dim, optimizer, boptim)
    node_features = nodes_out.cpu().detach().numpy()
    new_adj = edges_out.cpu().detach().numpy()
    i = i.cpu().detach().numpy()

    node_features = (node_features - node_b)/node_a
    node_features = qt.inverse_transform(node_features)

    #new_adj = np.where(new_adj>.5, 1.0 , 0)
    np.savetxt('./data/output/adj'+str(counter)+'.txt', new_adj)
    np.savetxt('./data/output/node'+str(counter)+'.txt',node_features)
    image = i*255.0
    image = np.reshape(image,(512,512,3))
    image = np.asarray(image, dtype=np.uint8)

    cv2.imwrite('./data/output/img'+str(counter)+'.png',image)
    counter += 1
    if counter == 6 :
        break


print(asd)
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
    
