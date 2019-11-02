import numpy as np 

img = np.zeros((16,16))

coor = []

pre_list = []
temp = []
#print(img.shape[0])
for i in range(img.shape[0]*img.shape[1]):
    temp.append(i)
    #print(temp)
    j = i +1 
    #print(j % 5)
    if j % img.shape[0] == 0:
        pre_list.append(temp)
        temp = []

#print(pre_list)

new = []


for i in range(len(pre_list)):
    temp = []
    for j in range(len(pre_list)):
        if j == 0 :
            temp.append([pre_list[i][j],pre_list[i][j+1]])
        elif j == len(pre_list)-1:
            temp.append([pre_list[i][j-1],pre_list[i][j]])
        else :
            temp.append([pre_list[i][j-1],pre_list[i][j],pre_list[i][j+1]])
    new.append(temp)
    temp = []

final = []
#print(new)
coun = 0
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        temp = []
        if i == 0 :
            one = new[i][j]
            two = new[i+1][j]
            temp.append(one+two)

        elif i == img.shape[0] -1 :
            one = new[i-1][j]
            two = new[i][j]
            temp.append(one+two)

        else :
            one = new[i-1][j]
            two = new[i][j]
            three = new[i+1][j]
            temp.append(one+two+three)
        
        temp = [item for sublist in temp for item in sublist]
        
        temp.remove(coun)
        #print(temp)
        for k in range(len(temp)):
            final.append([coun,temp[k]])

        coun += 1

print(final)
np.save('./data/numpy_arrays/adj.npy', final)
#print(np.load('./data/numpy_arrays/adj.npy'))