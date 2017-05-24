import numpy as np
import cv2
import os
import shutil
import time

###open feed and calculate center
cap = cv2.VideoCapture(0)
cap.open(0)
ret ,frame = cap.read()
time.sleep(3)
ret ,frame = cap.read()
center_x, center_y = int(cap.get(3)/2),int(cap.get(4)/2)


#*************TRAINING LABELS***************#
labels = np.array(["wall", "mug", "cube"])
#*******************************************#
###clean image data directories
if os.path.exists("./data/train"):
    shutil.rmtree("./data/train")
os.mkdir("./data/train")
if os.path.exists("./data/validation"):
    shutil.rmtree("./data/validation")
os.mkdir("./data/validation")

for x in np.nditer(labels):
    os.mkdir("./data/train/" + str(x))
    os.mkdir("./data/validation/" + str(x))


###misc vars
writeimgfiles = False
writecsvfiles = False
writetraindata = True
#f1 = open('pxldata.csv', 'w')
#f1 = open('data.csv', 'w')
#f2 = open('lbldata.csv', 'w')
status = "REC_OFF"
imgclass = 'NA'
t_or_v = 'train'
max = 1000
max_i = max

###video file output setup
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',cv2.cv.CV_FOURCC(*'mp4v'),20.0,(640,480))
#out = cv2.VideoWriter('output.avi',-1,20.0,(320,240))

i = 0
while(True):
    ret ,frame = cap.read()
    frame = cv2.resize(frame,(150,150),0,0)
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    imgarray = img.flatten()
    imgarray = ','.join(map(str, imgarray))
    #print("IMG: " + str(imgarray))
    print(str(img.shape))


    ####keypress listening####
    #start/stop recording to either CSV or image files
    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):
        if writetraindata:
            writetraindata = False
            t_or_v = 'validate'
            max_i = max / 5
        else:
            writetraindata = True
            t_or_v = 'train'
            max_i = max

    if key == ord('w'):
        if writecsvfiles:
            writecsvfiles = False
            status = "REC_OFF"
            i = 0
        else:
            writecsvfiles = True
            status = "REC_CSV"
    if key == ord('e'):
        if writeimgfiles:
            writeimgfiles = False
            status = "REC_OFF"
            i = 0
        else:
            writeimgfiles = True
            status = "REC_IMG"

    #change class labels
    if key == ord('1'):
        imgclass = labels[0]
    if key == ord('2'):
        imgclass = labels[1]
    if key == ord('3'):
        imgclass = labels[2]

    if writeimgfiles:

        if writetraindata:
            cv2.imwrite('./data/train/' + imgclass + '/' + imgclass + str(i) + '.png', frame)
        else:
            cv2.imwrite('./data/validation/' + imgclass + '/' + imgclass + str(i) + '.png', frame)
        #f2.write(str(label) + '\n')
        i = i + 1
        if i >= max_i:
            writeimgfiles = False
            status = "REC_OFF"
            i = 0

        #time.sleep(1)
    if writecsvfiles:
        #f1.write(str(imgarray) + '\n')
        f1.write(str(imgarray) + ',' + str(label) + '\n')
        #f2.write(str(label) + '\n')


    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(frame, str(status) + ', Frame: ' + str(i),(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    cv2.putText(frame, 'label: ' + imgclass,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    cv2.putText(frame, t_or_v,(5,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)

    final = np.concatenate((frame, img),axis=0)
    cv2.imshow('img',final)
    #cv2.moveWindow('img', 0,0)
    #out.write(frame)
    if key == ord('q'):
        break

cap.release()
#out.release()
f1.close()
f2.close()
cv2.destroyAllWindows()
