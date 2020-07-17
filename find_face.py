import cv2
import os
from glob import glob

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join("/content/PRNet/raw_images/"), files))
total_num = len(image_path_list)

for i, image_path in enumerate(image_path_list):

    name = image_path.strip().split('/')[-1][:-4]
    img = cv2.imread(image_path)
    #img = cv2.resize(img, (0, 0), fx = 0.3, fy = 0.3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (256, 256))
        cv2.imwrite("/content/PRNet/in/" + str(i)+".jpg", face)

print("DONE")

