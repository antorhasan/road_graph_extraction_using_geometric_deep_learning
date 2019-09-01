from os import listdir
from os.path import isfile, join
import networkx as nx 
import matplotlib.pyplot as plt
import cv2

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
    
    f = [f for f in listdir(path) if isfile(join(path, f))]
    #f = f[0:7]
    #print(f)
    for i in f :
        print(i)
        gph = open(path + i, 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont)
        #ls_node, ls_edge = gphtols(cont)
        make_gph(ls_node, ls_edge, range(len(ls_node))) 


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
    nx.draw(G, pos)
    plt.show()


def gphtols_view(graph):
    "convert .graph txt file to lists of nodes and edges and flip along horizontal axis"
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

view_gph('./data/gph_data/')