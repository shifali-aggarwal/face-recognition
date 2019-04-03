import os
import numpy as np
import cv2

BASE_DIR =os.path.dirname(os.path.abspath(__file__))
image_dir =os.path.join(BASE_DIR,'New folder')

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('.JPG') or file.endswith('.jpg') or file.endswith('.png'):
            path=os.path.join(root,file)
            face_cascade=cv2.CascadeClassifier('C:/Users/YATHINDRA RAO/Desktop/opencv/cascades/data/haarcascade_frontalface_alt2.xml')
            image=cv2.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi_image =image[y:y+h,x:x+w]#refion of intrest actually where the photo of the image stores
                img_item='my-image'+str(x)+str(w)+'.png'#saving the image
                cv2.imwrite(img_item,roi_image)
            print(faces)
#this code actually detects and crop the face of the person 
