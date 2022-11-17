from random import randrange
import cv2

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('takeoff.jpg')
#img = cv2.imread('Migo.jpg')
img = cv2.imread('awards.jpg')

grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), ( randrange(128, 256), randrange(128, 256),  randrange(128, 256)), 2)

#print(face_coordinates)



cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()

print("Code completed")