from os import listdir
from os.path import isfile, join
import networkx as nx 
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import numpy as np
import tensorflow as tf
#from preprocess import mean_std, change_range

tf.enable_eager_execution()

def path_sort(path):
    '''gets a path as input and returns a list of sorted filenames'''
    image_path = path
    img_lis = [f for f in listdir(image_path) if isfile(join(image_path, f))]
    for i in range(len(img_lis)):

        img_lis[i] = img_lis[i].split('_')

    '''sort the data'''
    for i in range(len(img_lis)):
        img_lis[i] = int(img_lis[i].split('.')[0])
    img_lis.sort()

    return img_lis
#path_sort('./data/gph_data/')

def view_gph(path):
    '''get path and view all graphs'''
    f = [f for f in listdir(path) if isfile(join(path, f))]
    #f = f[0:2000]
    #print(f)
    arr = []
    shob = []
    for i in f :
        print(i)
        gph = open(path + i, 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont)
        #ls_node, ls_edge = gphtols(cont)
        node = make_gph(ls_node, ls_edge, range(len(ls_node)))
        if node == 0 :
            continue
        #arr.append(node)
        #print(ls_node)
        for j in range(len(ls_node)):
            shob.append(ls_node[j])
            arr.append(ls_node[j][0])
    arr = np.asarray(arr)
    arr = arr+15000
    #shob = arr
    shob = np.asarray(shob)
    np.save('./data/numpy_arrays/nodes_attributes', shob)
    mean, std = mean_std(shob,'node_attr_np')
    new_data, a, b = change_range(shob,'node_attr_np')
    first = (shob-mean)/std
    plt.hist(first,bins=200)
    plt.show()
    second = a*first + b
    plt.hist(second,bins=200)
    plt.show()
    last = np.log(arr)
    plt.hist(last,bins=200)
    plt.show()


class create_gph():
    '''a graph is visualized from nodes,edges and position'''
    
    def __init__(self, nodes,edges,index):
        G = nx.Graph()
        counter = 0
        for i in index:
            G.add_node(i,coor=nodes[counter])
            counter += 1

        for i in range(len(edges)):
            G.add_edge(*edges[i])
        self.graph = G 

    def get_num_nodes(self):
        '''returns the number of nodes of the graph'''
        return self.graph.number_of_nodes()

    def get_adj(self):
        '''returns the adjacency matrix of the graph'''
        A = nx.adjacency_matrix(self.graph)
        return A.todense()

    
def make_gph(nodes, edges, index):
    '''a graph is visualized from nodes,edges and position'''
    G = nx.Graph()
    counter = 0
    for i in index:
        G.add_node(i,coor=nodes[counter])
        counter += 1

    for i in range(len(edges)):
        G.add_edge(*edges[i])

    pos = dict(zip(index, nodes))
    #print(index)
    #print(G.number_of_nodes())
    #A = nx.adjacency_matrix(G)
    #print(A.todense())
    #print(G.nodes())
    #nx.draw(G, pos)
    #plt.show()
    return G.number_of_nodes()


def gphtols_view(graph):
    "convert .graph txt file to lists of nodes and edges and does not flip along horizontal axis"
    ls_node = []
    ls_edge = []

    for i in range(len(graph)):

        if graph[i]!='\n':
            lis = graph[i].split()
            for j in range(len(lis)):
                if j == 1 :
                    lis[j] = float(lis[j])  
                else:
                    lis[j] = float(lis[j])
            ls_node.append(lis)
            #print(ls_node)
        else:
            var = i
            #print(var)
            break

    for j in range(var+1,len(graph)):
        lis = graph[j].split()
        #print(lis)
        for k in range(len(lis)):
            lis[k] = int(lis[k])
        ls_edge.append(tuple(lis))
        #print(*ls_edge[j])
    
    return ls_node,ls_edge

#view_gph('./data/test/gph/')
#view_gph('./data/gph_data/')
#print(np.load('./data/numpy_arrays/num_nodes.npy'))
#view_gph('./data/test/gph/')

if __name__ == "__main__":
    #view_gph('./data/test/sort/')
    #view_gph('./data/gph_data/')
    #rr = np.load('./data/numpy_arrays/num_nodes.npy')
    #print(np.amax(arr))
    #t = tf.constant([[1, 2, 3], [4, 5, 6]])
    
    #paddings = tf.constant([[0, 156-2], [0, 156-3]])
    # 'constant_values' is 0.
    # rank of 't' is 2.
    #print(tf.pad(t, paddings, "CONSTANT"))

    #print(np.load('./data/numpy_arrays/first/mean.npy'))
    #print(np.load('./data/numpy_arrays/node_attr_np/mean.npy'))
    path = './data/superimg/'
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_lis)):
        #print(path_lis[i])
        img = cv2.imread(path+path_lis[i])
        if img.shape[0]!=img.shape[1]:
            print(path_lis[i])
            
        #print(img.shape)
    pass