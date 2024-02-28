import streamlit as st
import cv2
from keras.models import load_model
import numpy as np
st.set_page_config(page_title="Mask Detection System",page_icon="https://cdn-icons-png.flaticon.com/128/4960/4960820.png")
st.title("Face Mask Detection System")

choice=st.sidebar.selectbox("Menu",("HOME","IP CAMERA","CAMERA"))
st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcScPo77ZlcJpGGjYOILiD8UdQQ9IisyHPTwghkmwMUCrGCAoY2Vb3Sf-5lI_m4wLbdy-a0&usqp=CAU")
if(choice=="HOME"):
    st.image("https://cdn.dribbble.com/users/1815739/screenshots/12127262/dribbble_facemask_v1.gif")
elif(choice=="IP CAMERA"):
    url=st.text_input("Enter IP Camera URL")
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        vid = cv2.VideoCapture(url)
        btn2=st.button("Stop Detection")
        if btn2:
            vid.release()
            st.experimental_rerun()
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
                window.image(frame,channels="BGR")
elif(choice=="CAMERA"):
    cam=st.selectbox("Choose 0 for primary camera and 1 for secondary camera",(0,1))
    btn=st.button("Start Detection")
    window=st.empty()
    if btn:
        try:
            vid = cv2.VideoCapture(cam)#0=primary camera oflaptop(webcam)/of dekstop computer(usb camera),1=Secondary camera of laptop(external usb camera) 
            btn2=st.button("Stop Detection")
            if btn2:
                vid.release()
                st.experimental_rerun()
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
                    window.image(frame,channels="BGR")
                else:
                    break
        except Exception as e:
            st.error(f"An error occurred: {e}")

        
    
    
