import numpy as np
import sys
import cv2
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from gph_crop import *
import matplotlib.pyplot as plt
#from util import write_gph, gphtols_view


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def createDataRecord(out_filename, addrs_y, img_path, gph_path):
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
        img_y = cv2.imread(img_path + str(addrs_y[i]))
        img_y = img_y/255
        img_y = np.asarray(img_y,dtype=np.float32) #all data has to be converted to np.float32 before writing
        
        gph = open(gph_path + addrs_y[i].split('.')[0] + '.txt', 'r')
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

def create_data(img_path,gph_path,dataset,split):
    '''create tfrecord from image and graph patches
    Args :
        - img_path : image directory
        - gph_path : graph directory
        - dataset : 'train' or 'test'. Also, serves as tfrecord name
        - split : percentage to split into
    '''
    
    trainY_list = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    #print(len(trainY_list))
    total_num = len(trainY_list)
    split_num = len(trainY_list)*split
    if dataset == 'train':
        path_list = trainY_list[0:split_num]
    elif dataset == 'test':
        path_list = trainY_list[split_num:total_num]
    print(len(path_list))
    #trainY_list = trainY_list[61440:76800]
    #createDataRecord('./data/record/'+ dataset +'.tfrecords', path_list, img_path, gph_path)
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
    
def dup_remove(input_dir, output_dir):
    '''remove duplicate node values and fix edge list accordingly
    Args :
        - input_dir : input directory of graphs to be fixed
        - output_dir : output directory where the graphs will be written
    '''
    path = input_dir
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_lis)):

        inputfilename = input_dir +path_lis[i]
        outputname = inputfilename.split('/')[-1]
        outputname = output_dir + outputname

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

def sort_latlon(input_dir,output_dir):
    '''sort node values from upper left corner and fix edge list accordingly
    Args :
        - input_dir : input directory of graphs to be fixed
        - output_dir : output directory where the graphs will be written
    '''
    path = input_dir
    path_lis = [f for f in listdir(path) if isfile(join(path, f))]
    for i in range(len(path_lis)):

        inputfilename = input_dir+path_lis[i]
    
        outputname = inputfilename.split('/')[-1]
        outputname = output_dir+ outputname
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

def fix_nodes(supimg_path,gph_path,output_dir,img_size):
    '''change node feature so that they are attributed based on the 
    center of each of the image crops instead of being attributed based
    on the center point of the superimage
    '''
    path = supimg_path
    path_list = [f for f in listdir(path) if isfile(join(path, f))]
    #path_list = path_list[0:1]

    gph_path = gph_path
    #gph_list = [f for f in listdir(path) if isfile(join(path, f))]
    #gph_list = gph_list[0:10]

    for i in range(len(path_list)):
        name = path_list[i].split('.')[0]
        print(name)
        img = cv2.imread(path + path_list[i],0)
        height = img.shape[0]
        width = img.shape[1]

        row_times = height/img_size
        column_times = width/img_size
        #print(row_times,column_times)
        first_center = [-(width/2)+(img_size/2), (height/2)-(img_size/2)]
        #print(first_center)
        for j in range(int(row_times)):
            for k in range(int(column_times)):
                gph_name = name + '_' + str(j) + '_' + str(k) + '.txt'
                #print(gph_name)
                gph = open(gph_path + gph_name, 'r')
                cont = gph.readlines()
                nodes, edges = gphtols_view(cont,False)
                #print(nodes)
                center = [first_center[0]+(img_size*k),first_center[1]-(img_size*j)]
                
                """ if k<3 and j<3 :
                    print(gph_name)
                    print(nodes[0:3]) """
                #print(center)

                for l in range(len(nodes)):
                    x_abs = abs(nodes[l][0] - center[0])
                    y_abs = abs(nodes[l][1] - center[1])

                    if nodes[l][0] > center[0] :
                        x = x_abs
                    else :
                        x = - x_abs
                    if nodes[l][1] > center[1]:
                        y = y_abs
                    else :
                        y = - y_abs
                    if nodes[l][0] == center[0]:
                        x = 0.0
                    if nodes[l][1] == center[1]:
                        y = 0.0
                    
                    nodes[l] = [x,y]
                    if x>img_size or x<-img_size or y>img_size or y<-img_size :
                        print(name)
                """ if k<3 and j<3 :
                    
                    print(center)
                    print(nodes[0:3]) """
                write_gph(output_dir+gph_name,nodes,edges)

def crop_fix():
    '''crop into 256 from original .graph txt file and fix all node and edge list'''
    crop_to_gph('./data/supergph/','./data/crop_graph/', True)
    dup_remove('./data/crop_graph/','./data/dup_rmgph/')
    sort_latlon('./data/dup_rmgph/','./data/sorted_gph/')
    fix_nodes('./data/superimg/','./data/sorted_gph/','./data/nodes_fixed/',256)

if __name__ == "__main__":
   
    #num_array()
    create_data('./data/img/','./data/nodes_fixed/','train',0.8)