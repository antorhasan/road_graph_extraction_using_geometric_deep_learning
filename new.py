from os import listdir
from os.path import isfile, join
import networkx as nx 
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import numpy as np
import tensorflow as tf
from preprocess import mean_std, change_range

#tf.enable_eager_execution()

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
    def __init__(self,path):


        '''plot graph from a text file
        argument : path to the text files
        output : matplotlib plot of the graphs in the folder
        '''
        f = [f for f in listdir(path) if isfile(join(path, f))]
        #f = f[0:3]
        #print(f)
        arr = []
        shob = []
        for i in f :
            print(i)
            gph = open(path + i, 'r')
            cont = gph.readlines()
            ls_node, ls_edge = gphtols_view(cont)
            make_graph(ls_node, ls_edge, range(len(ls_node)))

    
def view_gph(path):
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
        ls_node, ls_edge = gphtols_view(cont)
        if np.asarray(ls_node).any() > 256 or np.asarray(ls_node).any() < -256 :
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
    arr = np.asarray(arr)
    #arr = arr+15000
    #shob = arr
    shob = np.asarray(shob)
    np.save('./data/numpy_arrays/fixed_node.npy', shob)
    mean, std = mean_std(shob,'fixed_node')
    new_data, a, b = change_range(shob,'fixed_node')
    first = (shob-mean)/std
    plt.hist(arr,bins=200)
    plt.show()
    second = a*first + b
    plt.hist(second,bins=200)
    plt.show()
    last = np.log(arr)
    plt.hist(last,bins=200)
    plt.show()


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
        return A.todense()

class make_graph():

    def __init__(self, nodes, edges, index):
        '''create graph objects from nodes,edges and index list
        and plot the graph'''
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
        nx.draw(self.graph, self.positions)
        plt.show()
    
    def get_number_nodes(self):
        return self.graph.number_of_nodes()

    
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



if __name__ == "__main__":
    #view_graph('./data/temp/')
    #view_graph('./data/was/')
    view_gph('./data/nodes_fixed/')
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