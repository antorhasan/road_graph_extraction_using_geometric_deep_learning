import networkx as nx 
import matplotlib.pyplot as plt
from util import *
import cv2
from os import listdir
from os.path import isfile, join

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
    print(index)
    nx.draw(G, pos)
    plt.show()

def crop_to_gph(gph_path):
    '''crop graph txt according to given super img files'''

    f = [f for f in listdir(gph_path) if isfile(join(gph_path, f))]
    #f = f[0:2]

    for i in f :
        gph = open(gph_path + i, 'r')
        cont = gph.readlines()
        #print(len(cont))
        ls_node, ls_edge = gphtols(cont)
        name = i.split('.')[0]
        print(name)
        gph.close()
        #nodes, edges, index = gph_crop(ls_node, ls_edge, name)
        nodes, edges, index = crop(ls_node, ls_edge, name)

        make_gph(nodes, edges, index)
        write_gph('./data/try/'+ name +'.txt', nodes, edges)

fol_path = '../road_trc/dataset/data/graphs/'
f = [f for f in listdir(fol_path) if isfile(join(fol_path, f))]
f = f[0:2]
crop_to_gph(fol_path)

'''path = './data/supergph/'
f = [f for f in listdir(path) if isfile(join(path, f))]
f = f[0:2]

for i in f :
    print(i)
    gph = open(path + i, 'r')
    cont = gph.readlines()
    #print(len(cont))
    ls_node, ls_edge = gphtols(cont)
    new_node = []
    node_index = []
    for i in range(len(ls_node)):
        #if -x_len <= nodes[i][0] <= x_len and -y_len <= nodes[i][1] <= y_len : #this line is variable for area
        new_node.append(ls_node[i])
        node_index.append(i)
    
    new_edge = []
    for i in range(len(ls_edge)):
        if ls_edge[i][0] in node_index and ls_edge[i][1] in node_index:
            new_edge.append(ls_edge[i])
    make_gph(new_node, new_edge, node_index) '''



#print(len(edges))
#print(len(ls_node))

#print(G.edges.data())
#print(G.nodes.data())

