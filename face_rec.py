import cv2, sys, numpy, os
import time
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'


print('Make a normal face, calm down And Wait a second')
# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [numpy.array(lis) for lis in [images, labels]]

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

#use LBPHFace recognizer on camera frame
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
        #Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0),3)

        if prediction[1]<100:
           cv2.putText(im,'%s - %.0f - Status : Success' % (names[prediction[0]],prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
           a = ('%s' % (names)[prediction[0]])
           print("Your Name is >>> ",a)
           print("Status : \033[32mSuccessfully recognize face.\033[0m")
        else:
          cv2.putText(im,'Unknown - Status : Block',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
          print("Status : \033[31mFail to recognize face.\033[0m")
    cv2.imshow('OpenCV', im)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
