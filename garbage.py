import tensorflow as tf
from new import gphtols_view
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


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
    return new_data


path = "./data/gph_data/"

trainY_list = [f for f in listdir(path) if isfile(join(path, f))]
#trainY_list = trainY_list[0:20000]
print(len(trainY_list))

em = []

for i in range(len(trainY_list)):
    print(trainY_list[i])
    gph = open(path + trainY_list[i], 'r')
    cont = gph.readlines()
    ls_node, ls_edge = gphtols_view(cont)
    for j in range(len(ls_node)):
        em.append(ls_node[j])

em = np.asarray(em)

mean, std = mean_std(em, 'first')

em = (em-mean)/std
em = change_range(em,'first')

print(mean, std,em.shape)
arr = np.asarray(em[:,0:1])
plt.hist(em[:,1:2], bins=200)
plt.show()
plt.hist(em[:,0:1], bins=200)
plt.show()