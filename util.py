from os import listdir
from os.path import isfile, join
import cv2
def dirtodic(path):
    '''make a dict where the keys are common file name prefixes within a 
    directroy. input : path, output: dict obj of filenames'''

    dic = {}
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    
    for i in onlyfiles:
        for j in range(len(i)):
            if i[j] == '_' :
                var = i[0:j]
                break
        if var not in dic.keys() :
            dic.update({ var : [i] })
        else:
            dic[var].append(i)

    #print(dic.keys())
    return dic


print(len(dirtodic('/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery')))

var = dirtodic('/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery')

def mergeimg(lis):
    img_path = '/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery/'

    bos = sorted(lis['boston'])
    bos = bos[0:31]
    numlis = []
    #for i in bos:
    for i in bos :
        numbers = i.split('_')
        numlis.append(numbers[1])
    numlis = list(dict.fromkeys(numlis))
    print(numlis) 

    coun = 0
    coun_con = 0
    print(bos)
    prepre_ls = []
    for i in bos:
        splitted = i.split('_')

        if coun == 0 :
            pre_rnum = splitted[1]
            pre_cnum = splitted[2]
            pre_img = cv2.imread(img_path + i)
            #print(pre_rnum, pre_cnum)
            coun += 1
            continue

        new_rnum = splitted[1]
        new_cnum = splitted[2]
        new_img = cv2.imread(img_path + i)
        if int(new_rnum) == int(pre_rnum) :
            #print(new_cnum,pre_cnum)
            if int(new_cnum) > int(pre_cnum) :
                pre_img = cv2.vconcat([pre_img, new_img])
            else :
                pre_img = cv2.vconcat([new_img, pre_img])
            pre_rnum = new_rnum
            pre_cnum = new_cnum
            #prepre_num = new_rnum
            #print(prepre_num)
            #prepre_ls.append(prepre_num)
        elif int(new_rnum) != int(pre_rnum):
            if coun_con == 0 :
                pre_con_img = pre_img
                pre_rnum = new_rnum
                pre_cnum = new_cnum
                pre_img = cv2.imread(img_path + i)
                coun_con += 1
                continue

            new_con_img = pre_img
            print(numlis[coun_con],numlis[coun_con-1])
            #if int(pre_rnum) > int(prepre_num):
            print(coun_con)
            if int(numlis[coun_con]) > int(numlis[coun_con-1]):
                pre_con_img = cv2.hconcat([pre_con_img, new_con_img])
                print('h')
            else:
                pre_con_img = cv2.hconcat([new_con_img, pre_con_img])
                print('gd')
            pre_rnum = new_rnum
            pre_cnum = new_cnum
            pre_img = cv2.imread(img_path + i)
            coun_con += 1
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.imshow('image',pre_con_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            
            
        coun += 1

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',pre_con_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

    print(coun)
        
        
    return lis

mergeimg(var)

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