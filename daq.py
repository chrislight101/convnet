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
lbl = np.array(["cube", "mug", "none"])
#*******************************************#
###clean image data directories
deletedata = False
if deletedata:
    if os.path.exists("./data/train"):
        shutil.rmtree("./data/train")
    os.mkdir("./data/train")
    if os.path.exists("./data/validation"):
        shutil.rmtree("./data/validation")
    os.mkdir("./data/validation")

    for x in np.nditer(lbl):
        os.mkdir("./data/train/" + str(x))
        os.mkdir("./data/validation/" + str(x))

###misc vars
writeimgfiles = False
writetraindata = True
status = "REC_OFF"
imlbl = 'NA'
t_or_v = 'train'
max = 1000
max_i = max

i = 0
while(True):
    ret ,frame = cap.read()
    frame = cv2.resize(frame,(150,150),0,0)
    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    ####keypress listening####
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

    if key == ord('e'):
        if writeimgfiles:
            writeimgfiles = False
            status = "REC_OFF"
            i = 0
        else:
            writeimgfiles = True
            status = "REC_IMG"

    if key == ord('1'):
        imlbl = lbl[0]
    if key == ord('2'):
        imlbl = lbl[1]
    if key == ord('3'):
        imlbl = lbl[2]

    if writeimgfiles:
        if writetraindata:
            cv2.imwrite('./data/train/' + imlbl + '/' + imlbl + str(i) + '.png', frame)
        else:
            cv2.imwrite('./data/validation/' + imlbl + '/' + imlbl + str(i) + '.png', frame)
        i = i + 1
        if i >= max_i:
            writeimgfiles = False
            status = "REC_OFF"
            i = 0

    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(frame, str(status) + ', Frame: ' + str(i),(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    cv2.putText(frame, 'label: ' + imlbl,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    cv2.putText(frame, t_or_v,(5,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),2)
    final = np.concatenate((frame, img),axis=0)
    cv2.imshow('img',final)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
