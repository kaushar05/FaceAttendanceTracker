#Source code for Automated Attendance system through Face recognition.
import cv2
import numpy as np
import face_recognition
import os
import csv
from glob import glob
from datetime import datetime
    
#import ImageGrab from PIL
 
path = 'Training images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Encoding Function
def findEncodings(images):
    encodeList = []


    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


attended_students = []

def markAttendance(name):
        if name not in attended_students:
            attended_students.append(name)
            print(name)
            now = datetime.now()
            dtString, dstring= now.strftime('%Y-%b-%d'), now.strftime('%H:%M:%S')
            with open('Attendance.csv', 'a', newline='') as f:
                Inwriter = csv.writer(f)
                Inwriter.writerow([name,dtString,dstring])
                print(f'{name} added to Attendance.csv')
    

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDistance)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
# print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    key = cv2.waitKey(20) 
    if key == ord('q'):
        break
    if key == ord('p'): #for pausing the screen
        cv2.waitKey(-1) #wait until any key is pressed
        
#When everything done, release the capture      
cap.release()
cv2.destroyAllWindows()