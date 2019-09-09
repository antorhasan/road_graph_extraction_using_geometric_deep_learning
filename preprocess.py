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

    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs_y)):
        print(i)

        img_y = cv2.imread(path + str(addrs_y[i]))

        img_y = np.asarray(img_y)
        gph = open('./data/test/gph/' + addrs_y[i].split('.')[0] + '.txt', 'r')
        cont = gph.readlines()
        ls_node, ls_edge = gphtols_view(cont)
        if len(ls_node)==0 :
            continue
        node_attr = np.asarray(ls_node)
        print(node_attr)
        node_attr = (a*((node_attr - mean)/std))+b
        #print(node_attr)
        #ls_node, ls_edge = gphtols(cont)
        #node = make_gph(ls_node, ls_edge, range(len(ls_node)))
        graph = create_gph(ls_node, ls_edge, range(len(ls_node)))
        num_nodes = np.asarray(graph.get_num_nodes())
        adj_mtx = graph.get_adj()

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


path = "./data/test/img/"

trainY_list = [f for f in listdir(path) if isfile(join(path, f))]
#trainY_list = trainY_list[0:3]

#trainY = 

createDataRecord("./data/record/train.tfrecords", trainY_list)
#createDataRecord("./data/record/val.tfrecords", val_Y)
