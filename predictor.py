import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

###open feed and calculate center
cap = cv2.VideoCapture(0)
cap.open(0)
ret ,frame = cap.read()

model = load_model('model.h5')
model.load_weights('weights.h5')
label = "NA"

while(True):
    key = cv2.waitKey(1) & 0xFF
    ret,frame = cap.read()
    frame = cv2.resize(frame,(150,150),0,0)
    thresh = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(thresh,127,255,cv2.THRESH_BINARY)

    img = np.expand_dims(frame,axis=0)

    pred = model.predict_classes(img, verbose=1)
    print(pred)
    if (pred == 0):
        label = "cube"
    elif (pred == 1):
        label = "mug"
    elif (pred == 2):
        label = "wall"

    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.putText(frame, label,(5,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    final = np.concatenate((frame, thresh),axis=0)
    cv2.imshow('img',final)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
