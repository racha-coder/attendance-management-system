import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("TrainingImageLabels/Trainner.yml")
casecade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(casecade_path)
font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX

cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.2, 5)
    for x, y, w, h in faces:
        id, conf = recognizer.predict(gray_img[y : y + h, x : x + w])

        # cv2.rectangle(img, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(id), (x + 50, y - 10), font, 3, (255, 255, 255), 2)
        
    cv2.imshow("Image", img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()