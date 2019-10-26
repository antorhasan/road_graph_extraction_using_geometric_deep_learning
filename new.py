from os import listdir
from os.path import isfile, join
import networkx as nx 
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import numpy as np
import tensorflow as tf
from util import gphtols_view, crop_gph_256
from sklearn.preprocessing import QuantileTransformer


#tf.enable_eager_execution()



def change_range(data,folder):
    newmin = -1
    newmax = 1
    newR = newmax - newmin
    oldmin = np.amin(data)
    oldmax = np.amax(data)
    oldR = oldmax-oldmin
    a = newR / oldR
    b = newmin - ((oldmin*newR)/oldR)
    new_data = (data*a) + b
    np.save('./data/numpy_arrays/'+folder+'/a', a)
    np.save('./data/numpy_arrays/'+folder+'/b', b)
    return new_data,a,b


def mean_std(data, folder):
    '''given a numpy array, calculate and save mean and std'''
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    np.save('./data/numpy_arrays/'+folder+'/mean', mean)
    np.save('./data/numpy_arrays/'+folder+'/std', std)
    #print(data.shape)
    #print(mean.shape)
    #print(mean)
    #print(meam)
    #print(std.shape)
    return mean, std

def Neighbors(i, row, col):
    top = i-col
    bottom = i+col
    left = i-1
    right = i+1
    tl = top-1 #top left
    tr = top+1 #top right
    bl = bottom-1 #bottom left
    br = bottom+1 #bottom right

    if top < 0:
        top = -1
    if bottom > (row*col - 1):
        bottom = -1
    if left%col == (col-1):
        left = -1
    if right%col == 0:
        right = -1
    
    if top==-1 or left==-1:
        tl = -1
    if top==-1 or right==-1:
        tr = -1
    if bottom==-1 or left==-1:
        bl = -1
    if bottom==-1 or right==-1:
        br = -1

    temp = [tl,top,tr,left,i,right,bl,bottom,br]
    return temp

def lagbe_fix_adj():
    nplines = np.loadtxt("Input2.txt")
    row = len(nplines)
    col = len(nplines[0])

    outputfile = open("Output2.txt",'w+')
    npoutput = np.zeros(shape=(row*col, row*col))

    for i in range(row):
        for j in range(col):
            if nplines[i][j] == 0:
                iprime = i*col + j 
                for k in range(row*col):
                    npoutput[iprime][k] = 0
            elif nplines[i][j] == 1:
                currpos = i*col + j
                temp = Neighbors(currpos, row, col)
                res = []
                for k in temp:
                    if k!=-1:
                        tempr = int(k/col) 
                        tempc = k%col 
                        
                        if nplines[tempr][tempc] == 1:
                            res.append(k)
                            #print(k)
                iprime = i*col + j 
                for k in range(row*col):
                    npoutput[iprime][k] = 0
                    if k in res:
                        npoutput[iprime][k] = 1


    for i in range(len(npoutput)):
        s = ""
        for j in range(len(npoutput[0])):
            if npoutput[i][j] == 0:
                s += "0 "
            elif npoutput[i][j] == 1:
                s += "1 "

        outputfile.write(s)
        outputfile.write('\n')

    outputfile.close()

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

class view_graph():
    '''plot graph
    '''
    def __init__(self,path,flip):
        '''plot graph from a text file
        Args : 
            - path : to the text files
        Returns : 
            - matplotlib plot of the graphs in the folder
        '''
        f = [f for f in listdir(path) if isfile(join(path, f))]
        #f = f[0:3]
        #print(f)
        for i in f :
            print(i)
            gph = open(path + i, 'r')
            cont = gph.readlines()
            ls_node, ls_edge = gphtols_view(cont,flip)
            graph = make_graph(ls_node, ls_edge, range(len(ls_node)))
            graph.show_graph()
    
def view_normalize(path):
    '''get path and view all graphs'''
    f = [f for f in listdir(path) if isfile(join(path, f))]
    #f = f[0:2000]
    #print(f)
    arr = []
    shob = []
    for i in f :
        #print(i)
        gph = open(path + i, 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont,False)
        if np.asarray(ls_node).any() > 128 or np.asarray(ls_node).any() < -128 :
            print(i)
        #ls_node, ls_edge = gphtols(cont)
        #node = make_gph(ls_node, ls_edge, range(len(ls_node)))
        graph = make_graph(ls_node, ls_edge, range(len(ls_node)))
        node = graph.get_number_nodes()
        if node == 0 :
            continue

        #arr.append(node)
        #print(ls_node)
        for j in range(len(ls_node)):
            shob.append(ls_node[j])
            arr.append(ls_node[j][0])
    #arr = np.asarray(arr)
    #arr = arr+15000
    #shob = arr
    shob = np.asarray(shob)
    np.save('./data/numpy_arrays/nodes_out_128.npy', shob)
    plt.hist(shob,bins=200)
    plt.show()
    qt = QuantileTransformer(output_distribution='normal')
    shob = qt.fit_transform(shob)
    plt.hist(shob,bins=200)
    plt.show()
    #mean, std = mean_std(shob,'out_128')
    #shob = (shob-mean)/std
    #plt.hist(shob,bins=200)
    #plt.show()
    new_data, a, b = change_range(shob,'out_128')
    #first = (shob-mean)/std
    plt.hist(new_data,bins=200)
    plt.show()
    """ second = a*first + b
    plt.hist(second,bins=200)
    plt.show()
    last = np.log(arr)
    plt.hist(last,bins=200)
    plt.show() """

""" 
class create_gph():
    '''a graph is visualized from nodes,edges and position'''
    
    def __init__(self, nodes, edges, index):
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
        return A.todense() """

class make_graph():
    '''create a networkx graph object. methods include : plot the graph
    and get number of nodes of the graph'''

    def __init__(self, nodes, edges, index):

        '''create graph objects from nodes,edges and index list
        Args :
            - nodes : list of graph nodes
            - edges : list of graph edges
            - index : list of position of the graph nodes
        Properties :
            - positions : list of index of the graph
            - graph : networkx graph object
        '''
        G = nx.Graph()
        counter = 0
        for i in index:
            G.add_node(i,coor=nodes[counter])
            counter += 1

        for i in range(len(edges)):
            G.add_edge(*edges[i])

        pos = dict(zip(index, nodes))
        self.positions = pos
        self.graph = G
    
    def show_graph(self):
        '''plot the networkx graph aligning node positions'''
        nx.draw(self.graph, self.positions)
        plt.show()
    
    def get_number_nodes(self):
        '''return number of nodes of a graph
        Returns :
            - number of nodes in a graph as python int
        '''
        return self.graph.number_of_nodes()

    def get_graph(self):
        return self.graph

    def get_adj(self):
        '''returns the adjacency matrix of the graph'''
        A = nx.adjacency_matrix(self.graph)
        return A.todense()

    def remove_n(self, list):
        self.graph.remove_node(list)
    
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

def unknown():
    path = './data/superimg/'
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    num = 0
    gpath = './data/gph_data/'
    gpath_lis = [f for f in listdir(gpath) if isfile(join(gpath, f))]
    for i in range(len(gpath_lis)):
        gpath_lis[i] = gpath_lis[i].split('_')[0]
    
    for i in range(len(path_lis)):
        print(path_lis[i])
        name = path_lis[i].split('.')[0]
        img = cv2.imread(path+path_lis[i])
        #if img.shape[0]!=img.shape[1]:
        #    print(path_lis[i])
        one = img.shape[0]/256
        two = img.shape[1]/256
        element = one*two
        print(element)
        coun = 0
        for j in range(len(gpath_lis)):
            if name == gpath_lis[j]:
                coun +=1
        print(coun)
        #num = num+(one*two)
        #print(num)
        #print(img.shape)
    print(num)


def write_gph(path, nodes, edges):
    '''given nodes and edges list of a graph, it is written as txt'''
    with open(path, 'w') as f:
        for item in nodes:
            f.write("%s" % str(item[0]))
            f.write(" ")
            f.write("%s\n" % str(item[1]))
        f.write("\n")
        for e in edges:
            f.write("%s" % str(e[0]))
            f.write(" ")
            f.write("%s\n" % str(e[1]))


def crop_to_gph(gph_path, outdir, flip):
    '''crop graph txt according to given super img files
    Args :
        - gph_path : path of the supergraphs which needs cropping
        - outdir : output directory where the cropped graphs will be written
    '''

    f = [f for f in listdir(gph_path) if isfile(join(gph_path, f))]
    #f = f[0:7]

    for i in f :
        gph = open(gph_path + i, 'r')
        cont = gph.readlines()
        #print(len(cont))
        ls_node, ls_edge = gphtols_view(cont,flip)
        #ls_node, ls_edge = gphtols(cont)
        name = i.split('.')[0]
        print(name)
        #print(ls_edge)
        gph.close()
        #nodes, edges, index = gph_crop(ls_node, ls_edge, name)
        #nodes, edges, index = crop_p(ls_node, ls_edge, name)
        crop_gph_256(ls_node, ls_edge, name, outdir, 256)
        #make_gph(nodes, edges, index)
        #write_gph('./data/try/'+ name +'.txt', nodes, edges)



if __name__ == "__main__":
    view_normalize('./data/out_128/')
    #view_graph('./data/out_128/',False)
    #view_gph('./data/nodes_fixed/')
    #crop_to_gph('./data/supergph/','./data/crop_graph/', True)
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
    pass