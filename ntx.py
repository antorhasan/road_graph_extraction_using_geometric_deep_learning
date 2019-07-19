""" import networkx as nx 
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([0,1],time=[13,43])
G.add_edges_from([(0,1)])

plt.plot(graph)
plt.show()
nx.draw(G)
plt.show() """


try:
    print(int('_1'))
except:
    print('didn')


def mergeimg(lis):
    img_path = '/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery/'

    bos = sorted(lis['boston'])

    #boslis = []
    #for i in bos:
    coun = 0
    
    for i in bos:
        
        row_n = []
        if coun == 0 :
            srl = 0
            for j in range(len(i)):
                if i[j] == '_' :
                    srl +=1
                    if srl == 2 :
                        pre_name = i[0:j]
                        continue
        
        srl = 0
            for j in range(len(i)):
                if i[j] == '_' :
                    srl +=1
                    if srl == 2 :
                        pre_name = i[0:j]
                        continue

        if new_name == pre_name :
            srl = 0
            for j in range(len(i)):
                if i[j] == '_' :
                    srl +=1
                    if srl == 3 :
                        new_name = i[0:j]


    print(bos)

    return lis