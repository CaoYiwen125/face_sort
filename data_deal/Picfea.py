import numpy as np
import matplotlib.pyplot as plt
import cv2
start = 2417
end =5222
t=end - start
i=0
while i <= t:
     if i != 1228-start and i != 1232-start and i != 1808-start and i != 4056-start and i != 4136-start and i != 5004-start:
          with open('D:/Desktop/rawdata/'+str(start+i), 'rb') as fid:
               I = np.fromfile(fid, dtype=np.uint8)
               plt.imshow(np.reshape(I, (128, 128)), cmap='gray', interpolation='none')
               plt.savefig('d:/Desktop/newraw/'+str(start+i)+'.png')
               # plt.show()
     i+=1
# print(type(plt))

# with open('D:/Desktop/faceR', 'r') as fid:
#     data = fid.readline()
#     print(type(fid))
#     print(type(data))
#     print(data)

