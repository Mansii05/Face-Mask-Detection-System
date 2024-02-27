import cv2
from keras.models import load_model
import numpy as np
vid = cv2.VideoCapture("C:/Project/FaceMask/Scripts/group mask.mp4")
facemodel = cv2.CascadeClassifier("face.xml")
maskmodel=load_model("keras_mask.h5",compile=False)
i=1
while True:
    flag,frame = vid.read()
    if flag:
        pred=facemodel.detectMultiScale(frame)
        for (x,y,l,w) in pred:
            face_img=frame[y:y+w,x:x+l]
            face_img=cv2.resize(face_img,(224,224),interpolation=cv2.INTER_AREA)
            face_img=np.asarray(face_img,dtype=np.float32).reshape(1,224,224,3)
            face_img=(face_img/127.5)-1
            pred=maskmodel.predict(face_img)[0][1]
            if(pred>0.9):
                path="data_nomask/"+str(i)+".jpg"
                cv2.imwrite(path,frame[y:y+w,x:x+l])
                i=i+1
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,0,255),3)
            else:
                cv2.rectangle(frame,(x,y),(x+l,y+w),(0,255,0),3)
            
            
        cv2.namedWindow("mansi window",cv2.WINDOW_NORMAL)
        cv2.imshow("mansi window",frame)
        k=cv2.waitKey(30)
        if(k==ord("x")):
            break
    else:
        break
vid.release()
cv2.destroyAllWindows

