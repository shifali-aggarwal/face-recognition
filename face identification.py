import os
import numpy as np
import cv2
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade=cv2.CascadeClassifier('C:/Users/Innayat/Desktop/shifali/cascades/data/haarcascade_frontalface_alt2.xml')

#LBPH face recognizer

recognizer= cv2.face.LBPHFaceRecognizer_create()
x_train=[]
y_labels=[]
current_id=0
label_ids={}
for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith('.png') or file.endswith('.JPG') or file.endswith('.jpg'):
            path=os.path.join(root,file)
            label=os.path.basename(root).replace(" " ,"-").lower()#it give the name of the foler
            print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id=label_ids[label]
            print(label_ids)
            pil_image =Image.open(path).convert("L")#this converts image into grayscale
            image_array=np.array(pil_image,'uint8')#uint8 is type
            print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)
            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                y_labels.append(id)
                x_train.append(roi)

print(y_labels)
print(x_labels)
                
with open('label.pickle','wb'):
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(t_labels))
recognizer.save('trainer.yml')
    
