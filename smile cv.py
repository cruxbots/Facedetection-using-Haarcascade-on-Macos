



import cv2
import sys
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

faceCascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/4.1.0_2/share/opencv4/haarcascades/haarcascade_smile.xml')
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.9,
        minNeighbors=5
        
    )
    
    # Draw a rectangle around the faces
    def detect(gray, frame): 
        faces = face_cascade.detectMultiScale(gray, 1.9, 5) 
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,225),2)
            

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
