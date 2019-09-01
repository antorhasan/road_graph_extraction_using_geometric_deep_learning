import cv2

#img = cv2.imread('../road_trc/dataset/data/imagery/amsterdam_0_0_sat.png', 1)
img = cv2.imread('./data/superimg/amsterdam.png')
#print(im.shape)

""" for i in range(8):
    for j in range(8):

        im = img[512*i:512*(i+1),512*j:512*(j+1)]

       
        cv2.imwrite('./data/img/amsterdam'+str(i)+str(j)+'.png', im)
 """
for i in range(32):
    for j in range(32):

        im = img[256*i:256*(i+1),256*j:256*(j+1)]

       
        cv2.imwrite('./data/img/amsterdam_'+str(i)+'_'+str(j)+'.png', im)

""" cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows() """