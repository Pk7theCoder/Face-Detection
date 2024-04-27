import cv2  #importing Computer Vision Library
from random import randrange
#Load pre-trained data
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)

while True:
    succ_frame_read, frame = webcam.read()
    gs_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cords = trained_face_data.detectMultiScale(gs_img)  # detects the coordinates of the face

    # Drawing the rectangle over detected area
    for (x, y, w, h) in face_cords:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 255), randrange(128, 256)), 2)
    cv2.imshow("Face Detector", frame)  # Show image
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
webcam.release()