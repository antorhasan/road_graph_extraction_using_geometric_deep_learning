

def gphtols(graph):
    "convert .graph txt file to lists of nodes and edges"
    ls_node = []
    ls_edge = []

    for i in range(len(graph)):

        if graph[i]!='\n':
            lis = graph[i].split()
            for j in range(len(lis)):
                lis[j] = float(lis[j])
            ls_node.append(lis)
            #print(ls_node)
        else:
            var = i
            break

    for j in range(var+1,len(graph)):
        lis = graph[j].split()
        for k in range(len(lis)):
            lis[k] = int(lis[k])        
        ls_edge.append(tuple(lis))
    
    return ls_node,ls_edge