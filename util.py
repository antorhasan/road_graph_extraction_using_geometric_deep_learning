
def gph_crop(nodes, edges):
    "crops the graph to fit the image"
    new_node = []
    node_index = []
    for i in range(len(nodes)):
        if -4096.0 <= nodes[i][0] <= 4096.0 and -4096.0 <= nodes[i][1] <= 4096.0 : #this line is variable for area
            new_node.append(nodes[i])
            node_index.append(i)
    
    new_edge = []
    for i in range(len(edges)):
        if edges[i][0] in node_index and edges[i][1] in node_index:
            new_edge.append(edges[i])
    
    return new_node, new_edge, node_index


def gphtols(graph):
    "convert .graph txt file to lists of nodes and edges and flip along horizontal axis"
    ls_node = []
    ls_edge = []

    for i in range(len(graph)):

        if graph[i]!='\n':
            lis = graph[i].split()
            for j in range(len(lis)):
                if j == 1 :
                    lis[j] = -float(lis[j])  #negative, to flip the image with respect to horizontal axis
                else:
                    lis[j] = float(lis[j])
            ls_node.append(lis)
            #print(ls_node)
        else:
            var = i
            print(var)
            break

    for j in range(var+1,len(graph)):
        lis = graph[j].split()
        #print(lis)
        for k in range(len(lis)):
            lis[k] = int(lis[k])        
        ls_edge.append(tuple(lis))
        #print(*ls_edge[j])
    
    return ls_node,ls_edge