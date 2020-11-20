import cv2
import numpy as np
from collections import deque


# Basic Initialization 

def empty(a):
    print("")

bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
blpoints = [deque(maxlen=1024)]


blue_index = 0
green_index = 0 
red_index = 0
black_index = 0

kernel = np.ones((5,5),np.uint8)

frame_height = 640
frame_width = 480
cap =cv2.VideoCapture(0)
cap.set(3,frame_width)
cap.set(4,frame_height)
cap.set(10,150)

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]
colorIndex = 0


#Making Trackbar window and trackbars

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 560,320)
cv2.createTrackbar("Hue Min","Trackbars",0,131,empty)
cv2.createTrackbar("Hue Max","Trackbars",131,131,empty)
cv2.createTrackbar("Sat Min","Trackbars",150,255,empty)
cv2.createTrackbar("Sat Max","Trackbars",255,255,empty)
cv2.createTrackbar("Val Min","Trackbars",102,255,empty)
cv2.createTrackbar("Val Max","Trackbars",255,255,empty)


#Creating Paint Window

paintWin = np.zeros([480,640,3]) + 255
paintWin = cv2.rectangle(paintWin, (30,1), (120,65), (0,0,0) , 0 )
paintWin = cv2.rectangle(paintWin, (160,1) , (255,65) , colors[0], -1)
paintWin = cv2.rectangle(paintWin, (280,1) , (370,65) , colors[1], -1)
paintWin = cv2.rectangle(paintWin, (400,1) , (485,65) , colors[2], -1)
paintWin = cv2.rectangle(paintWin, (520,1) , (600,65) , colors[3], -1)

cv2.putText(paintWin, "CLEAR",(58,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 0), 1, cv2.LINE_AA)
cv2.putText(paintWin, "BLUE",(187,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWin, "GREEN",(295,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWin, "RED",(415,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
cv2.putText(paintWin, "BLACK",(530,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# Main Video Capturing 

while True:
    success,img = cap.read()
    img = cv2.flip(img,1)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # Getting Trackbar Positions for creating a mask

    h_min = cv2.getTrackbarPos("Hue Min","Trackbars")
    s_min = cv2.getTrackbarPos("Sat Min","Trackbars")
    v_min = cv2.getTrackbarPos("Val Min","Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max","Trackbars")
    s_max = cv2.getTrackbarPos("Sat Max","Trackbars")
    v_max = cv2.getTrackbarPos("Val Max","Trackbars")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])


    # Shows color Options in Video Outputed by Webcam 

    img = cv2.rectangle(img, (30,1), (120,65), (122,122,122) , -1 )
    img = cv2.rectangle(img, (160,1) , (255,65) , colors[0], -1)
    img = cv2.rectangle(img, (280,1) , (370,65) , colors[1], -1)
    img = cv2.rectangle(img, (400,1) , (485,65) , colors[2], -1)
    img = cv2.rectangle(img, (520,1) , (600,65) , colors[3], -1)

    cv2.putText(img, "CLEAR ALL",(58,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "BLUE",(187,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "GREEN",(295,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "RED",(415,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "BLACK",(530,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


    # Making A mask and dilating and eroding to remove impurities.  

    mask = cv2.inRange(imgHSV,lower,upper)
    mask = cv2.erode(mask,kernel,iterations=1)
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask = cv2.dilate(mask,kernel,iterations=1)


    cnt,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnt) > 0:
        # Sorting the contours on basis of their length
        cnt = sorted(cnt,key = cv2.contourArea, reverse=True)[0]
        # Finding the minimum area of contour
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        # Drawing the circle around the contour
        cv2.circle(img, (int(x),int(y)), int(radius), (0,255,255), 2)

        # Finding the area and centroid of the contour
        M = cv2.moments(cnt)
        center = (int(M['m10'] / M['m00']) , int(M['m01'] / M['m00']))


        # Drawing line on the screen
        if center[1] <= 65:
            if 30 <= center[0] <= 120: #Clear Button

                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                blpoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0 
                red_index = 0
                black_index = 0

                paintWin[67:,:,:] = 255
            elif 160 <= center[0] <= 255:
                    colorIndex = 0 # Blue
            elif 280 <= center[0] <= 370:
                    colorIndex = 1 # Green
            elif 400 <= center[0] <= 485:
                    colorIndex = 2 # Red
            elif 520 <= center[0] <= 600:
                    colorIndex = 3 # 
        else :
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                blpoints[black_index].appendleft(center)

    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        blpoints.append(deque(maxlen=512))
        black_index += 1

        # Draw lines of all the colors on the canvas and frame 
        points = [bpoints, gpoints, rpoints, blpoints]
        for i in range(len(points)):
            for j in range(len(points[i])):
                for k in range(1, len(points[i][j])):
                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue
                    cv2.line(img, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paintWin, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    cv2.imshow("Air Canvas",img)
    cv2.imshow("Mask",mask)
    cv2.imshow("Paint", paintWin)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

