import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
#cap.set(cv2.CAP_PROP_EXPOSURE, -2.0)
face_cascade = cv2.CascadeClassifier('C:/Users/Snorlax/Documents/Work 2018/Invent/Protoype/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/Snorlax/Documents/Work 2018/Invent/Protoype/haarcascade_eye.xml')
eyes = []
#Historgam

hist_height = 64
hist_width = 256
nbins = 32
bin_width = hist_width/nbins
mask = np.zeros((480,640),  np.uint8)
cv2.circle(mask,(240,320), 50, 255, -1)
j = np.zeros((hist_height,hist_width))
bins = np.arange(nbins,dtype=np.int32).reshape(nbins,1)
'''
facemask = np.zeros((480,640),  np.uint8)
facemask[430:480,0:50] = 255
cv2.imshow('facemask',facemask)

backmask1 = np.zeros((480,200), np.uint8)
backmask2 = np.zeros((480,200), np.uint8)
backmask[0:480,0:640] = 255
backmask[100:400,200:450] = 0
'''


while(True):
    ret,frame = cap.read()
    frame = cv2.flip(frame[0:480,0:640],1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faceimg = cv2.bitwise_and(frame,frame,mask = facemask)
    #backimg = cv2.bitwise_and(frame,frame,mask = backmask)
    faceval = cv2.mean(gray[150:350,250:400])
    cv2.imshow('bg',gray[90:120,0:50])
    bgval = cv2.mean(gray[90:120,0:50])
    #print(faceval,bgval)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    xcounter = 0
    ycounter = 0
    if (bgval[0]-faceval[0]) > 100:
        cv2.putText(frame,"turn around!",(250,250),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    for i, (x,y,w,h) in enumerate(faces):
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        xcenter = x+w/2
        ycenter = y+h/2
        xcounter+=xcenter
        ycounter+=ycenter
        if i == len(faces)-1:
            xcounter/=len(faces)
            ycounter/=len(faces)
            if xcounter > 350 or xcounter < 290:
                cv2.arrowedLine(frame,(320,240),(int(xcounter),int(ycounter)), (255,0,0),5)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist_hue = cv2.calcHist([hsv],[0],mask,[nbins],[0,256])
    cv2.normalize(hist_hue,hist_hue,hist_height,cv2.NORM_MINMAX)
    hist=np.int32(np.around(hist_hue))
    pts = np.column_stack((bins,hist))
    for x,y in enumerate(hist):
        cv2.rectangle(j,(int(x*bin_width),int(y)),(int(x*bin_width + bin_width-1),int(hist_height)),(255),-1)
    j=np.flipud(j)
    cv2.imshow('Color Histogram',j)
    j = np.zeros((hist_height,hist_width))

    if ret is True:
        cv2.imshow('frame', frame)

    else:
        continue
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
