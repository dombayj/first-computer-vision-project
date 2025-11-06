import cv2
import numpy as np
import face_recognition
import os
import pygame

pygame.mixer.init()
pygame.mixer.music.load("istiklal.mp3")

path = "photos"

images = []
classNames = []

myList = os.listdir(path=path)
for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(f"{os.path.splitext(cl)[0]}".upper())

def findEncodings(imgs):
    encodeList = []
    for img in imgs:
        imgD = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode = face_recognition.face_encodings(imgD)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding complete.")

cap = cv2.VideoCapture(0)
print("Camera initialized successfully.")

music_playing = False  # track whether music is playing

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_RGB2BGR)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    condition1 = False  

    for faceLoc, faceEncode in zip(facesCurFrame, encodeCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, faceEncode)
        faceDis = face_recognition.face_distance(encodeListKnown, faceEncode)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1, y2 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

            if name == "KARDY":
                condition1 = True

            


    
    if condition1  and not music_playing:
        pygame.mixer.music.play(-1)
        music_playing = True
    elif not condition1 and music_playing:
        pygame.mixer.music.stop()
        music_playing = False

    
    cv2.imshow("Webcam", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
