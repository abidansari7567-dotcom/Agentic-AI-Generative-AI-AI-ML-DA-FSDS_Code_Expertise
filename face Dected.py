import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier(r"D:\data science\AI\Gen AI\class\10th- Haar cascade classifier\Haarcascades\haarcascade_frontalface_default.xml")

image = cv2.imread(r"D:\data science\AI\Gen AI\project\OpenCv\abid.jpeg")


if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()
    
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face = face_classifier.detectMultiScale(gray, 1.3, 5)

if len(face) == 0:
    print("No faces found!")
else:
    for(x, y, w, h) in face:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)
        
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)  

    cv2.destroyAllWindows()
    