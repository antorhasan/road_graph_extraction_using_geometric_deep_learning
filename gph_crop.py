import networkx as nx 
import matplotlib.pyplot as plt
from util import *


gph = open('../road_trc/dataset/data/graphs/amsterdam.graph', 'r')
cont = gph.readlines()
#print(len(cont))
ls_node, ls_edge = gphtols(cont)

gph.close()

nodes, edges, index = gph_crop(ls_node, ls_edge)
#print(len(nodes))
#print(edges)
#print(len(index))

G = nx.Graph()
counter = 0
for i in index:
    #print(i)
    G.add_node(i,coor=nodes[counter])
    counter += 1

for i in range(len(edges)):
    #print(i)
    G.add_edge(*edges[i])

#print(len(edges))
#print(len(ls_node))

#print(G.edges.data())
#print(G.nodes.data())

pos = dict(zip(index, nodes))
#print(pos)
nx.draw(G, pos)
plt.show()