import cv2
import os
import numpy as np
from PIL import Image


recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def getImagesAndLabels(path):
    # Get the details of all the images in the directory
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    # create empty faces list
    faceSamples = []

    # create empty IDs list 
    ids = []
    
    # looping through all the image paths and loading the images and ids
    for imgPath in imagePaths:
        # Load the image and covert it into grayscale
        pilImage = Image.open(imgPath).convert('L')
        
        # Convert the PIL Image into numpy array of numbers
        image_np = np.array(pilImage, 'uint8')
        
        # Getting the ID from the image
        id = int(os.path.split(imgPath)[-1].split(".")[0])
        
        # Extract the face from the training sample 
        faces = detector.detectMultiScale(image_np)
        
        # If there is a face append that in the list along with the Id
        for (x, y, w, h) in faces:
            faceSamples.append(image_np[y:y+h, x:x+w])
            ids.append(id)
            
    return faceSamples, ids

faces, ids = getImagesAndLabels('TrainingImage') 
recognizer.train(faces, np.array(ids))
recognizer.save('TrainingImageLabels/Trainner.yml')