

gph = open('../road_trc/dataset/data/graphs/amsterdam.graph', 'r')
#print(gph.type())
cont = gph.readlines()
ls_node = []
ls_edge = []

for i in range(len(cont)):

    if cont[i]!='\n':
        lis = cont[i].split()
        for j in range(len(lis)):
            lis[j] = float(lis[j])
        ls_node.append(lis)
        #print(ls_node)
    else:
        var = i
        break

for j in range(var+1,len(cont)):
    lis = cont[j].split()
    for k in range(len(lis)):
        lis[k] = int(lis[k])        
    ls_edge.append(tuple(lis))



#print(ls_node)
#print(len(ls_node))
#print(ls_edge)
#print(var,len(cont))
gph.close()
#print(len(cont))
ls_edge = ls_edge[0:14]
ls_node = ls_node[0:10]
#print(ls_edge)
#print(ls_node)
import networkx as nx 
import matplotlib.pyplot as plt

G = nx.Graph()

#G.add_nodes_from(range(len(ls_node)), coor=ls_node)
#G.add_nodes_from([(0,{coor:[121,232]},(1,{coor:[121,232]},(2,{coor:[121,232]},(3,{coor:[121,232]},(4,{coor:[121,232]}])
#G.add_nodes_from([0], coor=[0,0])
#G.add_nodes_from([1], coor=[2,1])
print(G.nodes.data())
for i in range(len(ls_node)):
    #print(i)
    G.add_node(i,coor=ls_node[i])

G.add_edges_from(ls_edge)
print(G.edges.data())
print(G.nodes.data())
""" print(G.order())
var = range(10,G.order())
print(var)
print(G.edges.data())
print(G.nodes.data())
for i in var:
    G.remove_node(i)
print(G.nodes.data()) """
#d = dict(zip(range(2),[[12,3],[3,2]]))
#print(d)
""" print(ls_node)
print(ls_edge) """
pos = dict(zip(range(len(ls_node)),ls_node))
print(pos)
nx.draw(G, pos)
#nx.draw_networkx_edges(G)
#nx.draw_spectral(G)
#plt.plot(G)
plt.show()