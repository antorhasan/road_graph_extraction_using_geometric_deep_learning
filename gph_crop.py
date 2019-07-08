from util import *

gph = open('../road_trc/dataset/data/graphs/amsterdam.graph', 'r')
cont = gph.readlines()
#print(len(cont))
ls_node, ls_edge = gphtols(cont)

gph.close()

#ls_edge = ls_edge[0:14]
#ls_node = ls_node[0:10]
#print(ls_edge)
#print(ls_node)
import networkx as nx 
import matplotlib.pyplot as plt

G = nx.Graph()

#print(G.nodes.data())

for i in range(len(ls_node)):
    #print(i)
    G.add_node(i,coor=ls_node[i])

for i in range(len(ls_edge)):
    #print(i)
    G.add_edge(*ls_edge[i])

#print(len(ls_edge))
#print(len(ls_node))

#print(G.edges.data())
#print(G.nodes.data())

pos = dict(zip(range(len(ls_node)),ls_node))
print(pos)
nx.draw(G, pos)
plt.show()