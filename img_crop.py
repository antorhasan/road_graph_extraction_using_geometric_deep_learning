import cv2
#from new import path_sort
from os import listdir
from os.path import isfile, join

#img = cv2.imread('../road_trc/dataset/data/imagery/amsterdam_0_0_sat.png', 1)
image_path = './data/superimg/'
path = [f for f in listdir(image_path) if isfile(join(image_path, f))]
#path = path_sort('./data/superimg/')
#path = path[0:2]
print(path)
for i in range(len(path)):
    print(i)
    name = path[i].split('.')[0]
    img = cv2.imread('./data/superimg/'+str(path[i]))
    #print(im.shape)

    """ for i in range(8):
        for j in range(8):

            im = img[512*i:512*(i+1),512*j:512*(j+1)]

        
            cv2.imwrite('./data/img/amsterdam'+str(i)+str(j)+'.png', im)
    """
    for i in range(int(img.shape[0]/256)):
        for j in range(int(img.shape[1]/256)):

            im = img[256*i:256*(i+1),256*j:256*(j+1)]

        
            cv2.imwrite('./data/img/'+str(name)+'_'+str(i)+'_'+str(j)+'.png', im)

""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows() """