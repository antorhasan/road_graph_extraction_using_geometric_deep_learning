from os import listdir
from os.path import isfile, join
import cv2
import shapely      #needed for calculating intersections
from shapely.geometry import LineString, Point
#from gph_crop import write_gph
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

#var = dirtodic('/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery')

def mergeimg(lis):
    '''given a dict, everyimage from that dict is stiched into super images'''
    
    img_path = '/media/antor/Stuff/projects/road_net/code/road_trc/dataset/data/imagery/'

    for w in lis.keys() :

        bos = sorted(lis[w])
        #bos = bos[0:31]
        numlis = []
        #for i in bos:
        for i in bos :
            numbers = i.split('_')
            numlis.append(numbers[1])
        numlis = list(dict.fromkeys(numlis))
        

        coun = 0
        coun_con = 0
        print(bos)
        print(numlis)
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

                if i == bos[-1] :
                    new_con_img = pre_img
                    #if int(pre_rnum) > int(prepre_num):
                    print(numlis[coun_con],numlis[coun_con-1])
                    if int(numlis[coun_con]) > int(numlis[coun_con-1]):
                        pre_con_img = cv2.hconcat([pre_con_img, new_con_img])
                    else:
                        pre_con_img = cv2.hconcat([new_con_img, pre_con_img])

            elif int(new_rnum) != int(pre_rnum):
                if coun_con == 0 :
                    pre_con_img = pre_img
                    pre_rnum = new_rnum
                    pre_cnum = new_cnum
                    pre_img = cv2.imread(img_path + i)
                    coun_con += 1
                    continue

                new_con_img = pre_img
                #if int(pre_rnum) > int(prepre_num):
                print(numlis[coun_con],numlis[coun_con-1])
                if int(numlis[coun_con]) > int(numlis[coun_con-1]):
                    pre_con_img = cv2.hconcat([pre_con_img, new_con_img])
                
                else:
                    pre_con_img = cv2.hconcat([new_con_img, pre_con_img])
                    
                pre_rnum = new_rnum
                pre_cnum = new_cnum
                pre_img = cv2.imread(img_path + i)
                coun_con += 1
                
            coun += 1

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image',pre_con_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 

        cv2.imwrite('./data/superimg/'+ w +'.png', pre_con_img)

def crop_gph_256(nodes, edges, name):
    '''crops the graph to fit the image and also handles cropping through edge lines'''

    img_path = './data/data/superimg/'
    img = cv2.imread(img_path + name + '.png')
    
    x_len = float((img.shape[0])/2)
    y_len = float((img.shape[1])/2)

    for i in range(int(img.shape[0]/256)):
        
        for j in range(int(img.shape[1]/256)):

            '''boundary lines'''
            line1 = LineString([(-x_len+(j*256), y_len-(i*256)), (-x_len+((j+1)*256), y_len-(i*256))])
            line2 = LineString([(-x_len+((j+1)*256), y_len-(i*256)), (-x_len+((j+1)*256), y_len-((i+1)*256))])
            line3 = LineString([(-x_len+((j+1)*256), y_len-((i+1)*256)), (-x_len+(j*256), y_len-((i+1)*256))])
            line4 = LineString([(-x_len+(j*256), y_len-((i+1)*256)), (-x_len+(j*256), y_len-(i*256))])
            #print(-x_len+(j*256),y_len-(i*256),-x_len+((j+1)*256),y_len-((i+1)*256))
            lis_lines = [line1, line2, line3, line4]
            new_node = []
            node_index = []
            
            for k in range(len(edges)):
                node_1 = nodes[edges[k][0]]
                node_2 = nodes[edges[k][1]]

                '''conditions for checking if the nodes are in boundary'''
                cond_1 = (-x_len+(j*256)) <= node_1[0] <= (-x_len+((j+1)*256)) and (y_len-((i+1)*256)) <= node_1[1] <= (y_len-(i*256))
                cond_2 = (-x_len+(j*256)) <= node_2[0] <= (-x_len+((j+1)*256)) and (y_len-((i+1)*256)) <= node_2[1] <= (y_len-(i*256))
                #print(node_1,node_2,(-x_len+(j*256)),(-x_len+((j+1)*256)),(y_len-(i*256)),(y_len-((i+1)*256)))
                
                #print(cond_1,cond_2)
                #print('edge iteration')
                '''the three possibilities'''
                if (cond_1 == True) and (cond_2 == True) :

                    if node_1 not in new_node:
                        new_node.append(node_1)
                        node_index.append(edges[k][0])
                    if node_2 not in new_node:
                        new_node.append(node_2)
                        node_index.append(edges[k][1])

                    #print('both true')

                if (cond_1 == True) and (cond_2 == False) :
                    if node_1 not in new_node:
                        new_node.append(node_1)
                        node_index.append(edges[k][0])

                    line = LineString([tuple(node_1), tuple(node_2)])
                    for l in range(len(lis_lines)):
                        try:
                            int_pt = line.intersection(lis_lines[l])
                            point_of_intersection = int_pt.x, int_pt.y
                        except:
                            continue
                    new_node.append(list(point_of_intersection))
                    node_index.append(edges[k][1])

                    #print('one true')

                
                if (cond_1 == False) and (cond_2 == True) :
                    if node_2 not in new_node:
                        new_node.append(node_2)
                        node_index.append(edges[k][1])

                    line = LineString([tuple(node_1), tuple(node_2)])
                    for l in range(len(lis_lines)):
                        try:
                            int_pt = line.intersection(lis_lines[l])
                            point_of_intersection = int_pt.x, int_pt.y
                        except:
                            continue
                    new_node.append(list(point_of_intersection))
                    node_index.append(edges[k][0])

                    #print('two true')
            
            '''updating the edge list according to new nodes list'''
            new_edge = []
            for k in range(len(edges)):
                if edges[k][0] in node_index and edges[k][1] in node_index:
                    new_edge.append(edges[k])
            
            dic_in = {}
            '''updating node index to start from zero'''
            for k in range(len(node_index)):
                dic_in.update({node_index[k]:k})

            '''updating edgelist using node index'''
            ed = []
            for k in range(len(new_edge)):
                a = dic_in[new_edge[k][0]]
                b = dic_in[new_edge[k][1]]
                ed.append(tuple([a,b]))
        
            write_gph('./data/graph/'+ name +'_'+str(i)+'_'+str(j)+'.txt', new_node, ed)

        print(i)


def crop_p(nodes, edges, name):
    '''crops the graph to fit the image and also handles cropping through edge lines'''

    img_path = './data/superimg/'
    img = cv2.imread(img_path + name + '.png')
    
    x_len = float((img.shape[0])/2)
    y_len = float((img.shape[1])/2)

    '''boundary lines'''
    line1 = LineString([(-x_len, -y_len), (x_len, -y_len)])
    line2 = LineString([(-x_len, -y_len), (-x_len, y_len)])
    line3 = LineString([(-x_len, y_len), (x_len, y_len)])
    line4 = LineString([(x_len, y_len), (x_len, -y_len)])

    lis_lines = [line1, line2, line3, line4]
    new_node = []
    node_index = []

    for i in range(len(edges)):
        node_1 = nodes[edges[i][0]]
        node_2 = nodes[edges[i][1]]

        '''conditions for checking if the nodes are in boundary'''
        cond_1 = -x_len <= node_1[0] <= x_len and -y_len <= node_1[1] <= y_len
        cond_2 = -x_len <= node_2[0] <= x_len and -y_len <= node_2[1] <= y_len

        '''the three possibilities'''
        if (cond_1 == True) and (cond_2 == True) :
            if node_1 not in new_node:
                new_node.append(node_1)
                node_index.append(edges[i][0])
            if node_2 not in new_node:
                new_node.append(node_2)
                node_index.append(edges[i][1])

        if (cond_1 == True) and (cond_2 == False) :
            if node_1 not in new_node:
                new_node.append(node_1)
                node_index.append(edges[i][0])

            line = LineString([tuple(node_1), tuple(node_2)])
            for j in range(len(lis_lines)):
                try:
                    int_pt = line.intersection(lis_lines[j])
                    point_of_intersection = int_pt.x, int_pt.y
                except:
                    continue
            new_node.append(list(point_of_intersection))
            node_index.append(edges[i][1])

        
        if (cond_1 == False) and (cond_2 == True) :
            if node_2 not in new_node:
                new_node.append(node_2)
                node_index.append(edges[i][1])

            line = LineString([tuple(node_1), tuple(node_2)])
            for j in range(len(lis_lines)):
                try:
                    int_pt = line.intersection(lis_lines[j])
                    point_of_intersection = int_pt.x, int_pt.y
                except:
                    continue
            new_node.append(list(point_of_intersection))
            node_index.append(edges[i][0])
    
    '''updating the edge list according to new nodes list'''
    new_edge = []
    for i in range(len(edges)):
        if edges[i][0] in node_index and edges[i][1] in node_index:
            new_edge.append(edges[i])
    
    dic_in = {}
    '''updating node index to start from zero'''
    for i in range(len(node_index)):
        dic_in.update({node_index[i]:i})

    '''updating edgelist using node index'''
    ed = []
    for i in range(len(new_edge)):
        a = dic_in[new_edge[i][0]]
        b = dic_in[new_edge[i][1]]
        ed.append(tuple([a,b]))


    return new_node, ed, range(len(node_index))


def crop(nodes, edges, name):
    "crops the graph to fit the image"
    img_path = './data/superimg/'
    img = cv2.imread(img_path + name + '.png')

    x_len = float((img.shape[0])/2)
    y_len = float((img.shape[1])/2)

    #print(x_len,y_len)

    new_node = []
    node_index = []
    for i in range(len(nodes)):
        if -x_len <= nodes[i][0] <= x_len and -y_len <= nodes[i][1] <= y_len : #this line is variable for area
            new_node.append(nodes[i])
            node_index.append(i)
    
    new_edge = []
    for i in range(len(edges)):
        if edges[i][0] in node_index and edges[i][1] in node_index:
            new_edge.append(edges[i])
    
    dic_in = {}

    for i in range(len(node_index)):
        dic_in.update({node_index[i]:i})
    #print(dic_in)

    #print(new_edge)
    ed = []
    for i in range(len(new_edge)):
        #for j in range(len(i)):
        a = dic_in[new_edge[i][0]]
        b = dic_in[new_edge[i][1]]
        ed.append(tuple([a,b]))

    #print(ed)

    return new_node, ed, range(len(node_index))



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

def gphtols_view(graph):
    '''convert .graph txt file to lists of nodes and edges, does not flip along horizontal axis
    and ready it for further processing so that it can viewed'''
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

if __name__ == "__main__":
    crop_gph_256
    pass