import numpy as np
import sys
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from new import *
import matplotlib.pyplot as plt



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createDataRecord(out_filename, addrs_y):
    mean = np.load('./data/numpy_arrays/first/mean.npy')
    std = np.load('./data/numpy_arrays/first/std.npy')
    a = np.load('./data/numpy_arrays/first/a.npy')
    b = np.load('./data/numpy_arrays/first/b.npy')

    an = np.load('./data/numpy_arrays/nodes/a.npy')
    bn = np.load('./data/numpy_arrays/nodes/b.npy')
    
    #num_n = []
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(i)
        if i == 0 :
            print(addrs_y[i])
        img_y = cv2.imread('./data/img/' + str(addrs_y[i]))
        img_y = img_y/255
        img_y = np.asarray(img_y,dtype=np.float32) #all data has to be converted to np.float32 before writing
        
        gph = open('./data/final_gph/' + addrs_y[i].split('.')[0] + '.txt', 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont)
        if i == 0 :
            print(ls_node)
        if len(ls_node)==0 :
            continue
        #node_attr = np.asarray(ls_node,dtype=np.float32)
        #print(ls_node)
        node_attr = (a*((ls_node - mean)/std))+b
        node_attr = np.asarray(node_attr,dtype=np.float32)
        #print(node_attr)
        #ls_node, ls_edge = gphtols(cont)
        #node = make_gph(ls_node, ls_edge, range(len(ls_node)))
        graph = create_gph(ls_node, ls_edge, range(len(ls_node)))
        num_nodes = graph.get_num_nodes()
        num_nodes = np.log(num_nodes)
        num_nodes = (an*num_nodes)+bn

        num_nodes = np.asarray(num_nodes,dtype=np.float32)
        adj_mtx = graph.get_adj()
        adj_mtx = np.asarray(adj_mtx,dtype=np.float32)
        #num_n.append(num_nodes)

        if i == 0 :
            print(node_attr,node_attr.shape)
            print(adj_mtx)
            print(num_nodes)
            print(img_y)

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

    #print(np.amax(num_n),np.amin(num_n))

def create_data():
    #path = "./data/test/img/"   #test data only amsterdam
    path = "./data/img/"        #full data

    trainY_list = [f for f in listdir(path) if isfile(join(path, f))]

    #trainY_list = trainY_list[0:61440]
    trainY_list = trainY_list[61440:76800]

    #trainY = 

    createDataRecord("./data/record/train_15.tfrecords", trainY_list)
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
    #path = './data/test/gph/'
    path = './data/gph_data/'
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_lis)):

        inputfilename = "./data/gph_data/"+path_lis[i]
        outputname = inputfilename.split('/')[-1]
        outputname = './data/mod_gph/'+ outputname

        outputfile = open(outputname,'w+')

        duplicate_removal(inputfilename,outputfile)

        outputfile.close()



def sorting_latlng(inputfilename,outputfile):
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
    lnglat = sorted(lnglat, key=lambda l:l[1], reverse=True) #sorted on the basis of latitude
    #print(lnglat)
    #print('------------------------------------------------')

    i = 0
    j = 0
    while i < (len(lnglat)-1):
        if lnglat[i][1] == lnglat[i+1][1]:
            j = i+1
            while j < (len(lnglat)-1):
                if lnglat[j][1] != lnglat[j+1][1]:
                    break 
                j += 1

            temp = lnglat[i:j+1]
            temp = sorted(temp, key=lambda l:l[0])
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

def sort_latlon():
    #input file to be sorted
    #path = './data/test/mod/'
    path = './data/mod_gph/'
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_lis)):

        inputfilename = "./data/mod_gph/"+path_lis[i]
    
        outputname = inputfilename.split('/')[-1]
        outputname = './data/final_gph/'+ outputname
        outputfile = open(outputname, 'w+')


        sorting_latlng(inputfilename,outputfile)
        outputfile.close()


def mean_std(data, folder):
    '''given a numpy array, calculate and save mean and std'''
    data = np.asarray(data)
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    np.save('./data/numpy_arrays/'+folder+'/mean', mean)
    np.save('./data/numpy_arrays/'+folder+'/std', std)
    #print(data.shape)
    #print(mean.shape)
    #print(mean)
    #print(meam)
    #print(std.shape)
    return mean, std

def change_range(data,folder):
    newmin = -1
    newmax = 1
    newR = newmax - newmin
    oldmin = np.amin(data)
    oldmax = np.amax(data)
    oldR = oldmax-oldmin
    a = newR / oldR
    b = newmin - ((oldmin*newR)/oldR)
    new_data = (data*a) + b
    np.save('./data/numpy_arrays/'+folder+'/a', a)
    np.save('./data/numpy_arrays/'+folder+'/b', b)
    return new_data,a,b

def num_array():
    '''save mean,std,a,b of number of nodes of the graphs'''
    
    arr = np.load('./data/numpy_arrays/num_nodes.npy')
    plt.hist(arr, bins=200)
    plt.show()
    print(arr)
    #arr = [x for x in arr if x < 60]
    arr = np.log(arr)
    plt.hist(arr, bins=200)
    plt.show()
    #mean, std = mean_std(arr,'nodes')
    new,a,b = change_range(arr,'nodes')
    #print(mean,std,a,b)
    #arr = (arr-mean)/std
    #plt.hist(arr, bins=200)
    #plt.show()
    arr = (a*arr)+b
    plt.hist(arr, bins=200)
    plt.show()

def fix_nodes():
    '''change node feature so that they are attributed based on the 
    center of each of the image crops instead of being attributed based
    on the center point of the superimage
    '''
    path = "./data/superimg/"
    path_list = [f for f in listdir(path) if isfile(join(path, f))]
    path_list = [0:1]

    gph_path = './data/final_gph/'
    gph_list = [f for f in listdir(path) if isfile(join(path, f))]
    gph_list = [0:10]

    for i in range(len(path_list)):
        name = path_list[i].split('.')[0]

        img = cv2.imread(path + path_list[i],0)
        height = img.shape[0]
        width = img.shape[1]

        row_times = height/256
        column_times = width/256

        first_center = [-(width/2)+128, (height/2)-128)]
        for j in range(row_times):
            for k in range(column_times):
                
                
                nodes, edges = gphtols_view()



if __name__ == "__main__":
    #dup_remove()
    #sort_latlon()
    #num_array()
    #create_data()