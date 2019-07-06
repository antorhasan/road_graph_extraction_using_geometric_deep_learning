import networkx as nx 
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([0,1],time=[13,43])
G.add_edges_from([(0,1)])

""" plt.plot(graph)
plt.show() """
nx.draw(G)
plt.show()