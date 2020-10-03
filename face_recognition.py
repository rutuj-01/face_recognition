!pip3 install face_recognition
from imutils import paths
import dlib
import os
import cv2
import face_recognition
from PIL import Image
import numpy as np
import pickle

#!ls
#!pwd
#!face_recognition --help
images = os.listdir('/content/drive/My Drive/images')
#!cat /proc/cpuinfo
image = Image.open('/content/drive/My Drive/gg.jpg')

image_to_be_matched = face_recognition.load_image_file('/content/drive/My Drive/12335555.jpeg')

image_to_be_matched_encoded=face_recognition.face_encodings(image_to_be_matched)[0]

#!%cd '{PATH}'
#!find '.' -name '*.ipynb_checkpoints' -exec rm -r {} +
from IPython.display import Image


for image in images:   
    
    current_image=face_recognition.load_image_file("/content/drive/My Drive/images/" + image)
    current_image_encoded = face_recognition.face_encodings(current_image)[0]
    result=face_recognition.compare_faces([image_to_be_matched_encoded],current_image_encoded,tolerance=0.55)
    print(result)
    from IPython.display import Image, display
    
    if(result[0]==True):
      
      print("Matched "+image)
      display(Image('/content/drive/My Drive/images/' + image , width=100,height=100))
