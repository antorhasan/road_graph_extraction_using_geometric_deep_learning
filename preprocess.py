import numpy as np
import sys
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from new import *



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createDataRecord(out_filename, addrs_y):
    mean = np.load('./data/numpy_arrays/first/mean.npy')
    std = np.load('./data/numpy_arrays/first/std.npy')
    a = np.load('./data/numpy_arrays/first/a.npy')
    b = np.load('./data/numpy_arrays/first/b.npy')
    num_n = []
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        #print(i)
        if i == 0 :
            print(addrs_y[i])
        img_y = cv2.imread(path + str(addrs_y[i]))
        img_y = img_y/255
        img_y = np.asarray(img_y,dtype=np.float32)
        
        gph = open('./data/test/gph/' + addrs_y[i].split('.')[0] + '.txt', 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont)
        if i == 0 :
            print(ls_node)
        if len(ls_node)==0 :
            continue
        node_attr = np.asarray(ls_node,dtype=np.float32)
        #print(node_attr)
        node_attr = (a*((node_attr - mean)/std))+b
        #print(node_attr)
        #ls_node, ls_edge = gphtols(cont)
        #node = make_gph(ls_node, ls_edge, range(len(ls_node)))
        graph = create_gph(ls_node, ls_edge, range(len(ls_node)))
        num_nodes = np.asarray(graph.get_num_nodes(),dtype=np.float32)
        adj_mtx = graph.get_adj()
        adj_mtx = np.asarray(adj_mtx,dtype=np.float32)
        num_n.append(num_nodes)

        if i == 0 :
            print(node_attr,node_attr.shape)
            print(adj_mtx)

        feature = {
            'image_y': _bytes_feature(img_y.tostring()),
            'gph_nodes': _bytes_feature(node_attr.tostring()),
            'gph_adj' : _bytes_feature(adj_mtx.tostring()),
            'gph_node_num' : _bytes_feature(num_nodes.tostring())
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()

    print(np.amax(num_n),np.amin(num_n))

def create_data():
    path = "./data/test/img/"

    trainY_list = [f for f in listdir(path) if isfile(join(path, f))]

    trainY_list = trainY_list[0:3]

    #trainY = 

    createDataRecord("./data/record/train.tfrecords", trainY_list)
    #createDataRecord("./data/record/val.tfrecords", val_Y)


def duplicate_removal(filename,outputfile):
    lines = []
    with open(filename) as f:
        lines = f.readlines()
        
    lines = [x.strip() for x in lines]
    flag = []
    garbage = []

    for i in range(len(lines)):
        if lines[i] == "":
            break 
        flag.append(0)
        if lines[i] == lines[i+1]:
            flag[i] = 1
            garbage.append(i)

    flag.append(0)

    features = []
    i = 0
    for i in range(len(lines)):
        if lines[i] == "":
            break 
        
        if flag[i] == 0:
            features.append(lines[i])


    #update flag list
    last = 0
    for j in range(len(flag)):
        if flag[j] == 1:
            last += 1
        flag[j] = last 


    edges = []
    i += 1
    while i < len(lines):
        nums = lines[i].split()
        n1 = int(nums[0])
        n2 = int(nums[1])
        
        if n1 in garbage or n2 in garbage:
            i += 1
            continue

        n1 = n1 - flag[n1]
        n2 = n2 - flag[n2]

        line = str(n1) + ' ' + str(n2)
        edges.append(line)

        i += 1

    for l in features:
        outputfile.write(l + "\n")
    outputfile.write("\n")

    for l in edges:
        outputfile.write(l + "\n")
    
def dup_remove():
    #give input file path here
    inputfilename = "./data/test/gph/amsterdam_0_0.txt"
    outputname = inputfilename.split('/')[-1]
    outputname = './data/test/mod/'+ outputname

    outputfile = open(outputname,'w+')

    duplicate_removal(inputfilename,outputfile)

    outputfile.close()

def sorting_latlng(inputfilename):
    with open(inputfilename) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]

    #lat = []
    #lng = []
    lnglat = []
    savedlnglat = []
    newEdges = []

    edgenum = 0
    for l in lines:
        if l == "":
            break
        nums = l.split(' ')
        n1 = float(nums[0])
        n2 = float(nums[1])
        #lng.append(n1)
        #lat.append(n2)
        lst = [n1,n2]
        lnglat.append(lst)
        savedlnglat.append(lst)
        edgenum += 1

    edgenum += 1

    #sorting
    lnglat = sorted(lnglat, key=lambda l:l[0]) #sorted on the basis of longitude
    #print(lnglat)
    #print('------------------------------------------------')

    i = 0
    j = 0
    while i < (len(lnglat)-1):
        if lnglat[i][0] == lnglat[i+1][0]:
            j = i+1
            while j < (len(lnglat)-1):
                if lnglat[j][0] != lnglat[j+1][0]:
                    break 
                j += 1

            temp = lnglat[i:j+1]
            temp = sorted(temp, key=lambda l:l[1], reverse=True)
            lnglat[i:j+1] = temp
            i = j 

        i+=1

    #print((lnglat))

    #Edge correction according to new sorted order
    for i in range(edgenum, len(lines), 1):
        nums = lines[i].split(' ')
        n1 = int(nums[0])
        n2 = int(nums[1])

        ind1 = lnglat.index(savedlnglat[n1])
        ind2 = lnglat.index(savedlnglat[n2])

        lst = [ind1, ind2]
        newEdges.append(lst)

    
    #printing in output file
    for l in lnglat:
        s = str(l[0]) + " " + str(l[1]) + "\n"
        outputfile.write(s)
    
    outputfile.write("\n")

    for l in newEdges:
        s = str(l[0]) + " " + str(l[1]) + "\n"
        outputfile.write(s)
    


#input file to be sorted
inputfilename = './data/test/mod/amsterdam_0_0.txt'

outputname = 'SORTED' + inputfilename.split('/')[-1]
outputfile = open(outputname, 'w+')

sorting_latlng(inputfilename)
outputfile.close()


