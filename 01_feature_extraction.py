# Import Necessary Packages
import numpy as np
import math as m
import cv2
import csv
import time

# Create CSV file
header = ['timer', 'len_bottom', 'len_upper', 'len_diff', 'peak_bottom', 'peak_upper', 'peak_diff', 'joint_angle'] 
        # 'angle0', 'angle1', 'angle2'
csvfile = open('chopsticks01.csv', 'w')
writer = csv.writer(csvfile, delimiter = ',', lineterminator='\n')
writer.writerow(header)

start_time = time.time()

# Load_Video
PATH = r"C:\User_Folder\cmt_training.mp4"

capture = cv2.VideoCapture(PATH)

if not capture.isOpened():
    print("Cannot open camera")
    exit()

peak_bottom = 0
peak_upper = 0
peak_diff = 0

peak_bottom_get = 0
peak_upper_get = 0
peak_diff_get = 0

while True:

    #Global Capture
    len_bottom = 0
    len_upper = 0
    angle = 0

    # Capture Frame
    ret, frame = capture.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Scale Down the Frame to 50%
    scale_percent = 50 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    # Copy and Convert RGB to HSV
    image = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ############################################################### Detect Red Marker

    red_lower = np.array([0,175,100])
    red_upper = np.array([10,255,255])
    mask = cv2.inRange(hsv, red_lower, red_upper)

    mask = cv2.medianBlur(mask, 15)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=5)
    #cv2.imshow('mask', mask)

    # Detect ROI countours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 0

    red_ROI_number = 0
    red_x = []
    red_y = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (36,255,12), 2)
        red_ROI_number += 1
        red_x.append(x + int(w/2)) 
        red_y.append(y + int(h/2))

    if red_ROI_number == 4:
        
        # Draw Feature and Line
        feature_0 = (red_x[0], red_y[0])
        feature_1 = (red_x[1], red_y[1])
        feature_2 = (red_x[2], red_y[2])
        feature_3 = (red_x[3], red_y[3])
        
        # Get Chopsticks Len
        len_0 = m.sqrt( (red_x[0]-red_x[1])**2 + (red_y[0]-red_y[1])**2 )
        len_1 = m.sqrt( (red_x[1]-red_x[2])**2 + (red_y[1]-red_y[2])**2 )
        len_2 = m.sqrt( (red_x[2]-red_x[3])**2 + (red_y[2]-red_y[3])**2 )
        len_3 = m.sqrt( (red_x[3]-red_x[0])**2 + (red_y[3]-red_y[0])**2 )
        len_4 = m.sqrt( (red_x[0]-red_x[2])**2 + (red_y[0]-red_y[2])**2 )
        len_5 = m.sqrt( (red_x[1]-red_x[3])**2 + (red_y[1]-red_y[3])**2 )

        # Order Chopsticks Len
        len_order = [ len_0, len_1, len_2, len_3, len_4, len_5 ]
        chopsticks_len = sorted(len_order, reverse=True)[2]

        # Get Red Order
        red_order = [ red_x[0], red_x[1], red_x[2], red_x[3] ]
        
        bottom_0 = red_order.index( sorted(red_order, reverse=True)[3] )
        bottom_1 = red_order.index( sorted(red_order, reverse=True)[2] )
        upper_0 = red_order.index( sorted(red_order, reverse=True)[1] )
        upper_1 = red_order.index( sorted(red_order, reverse=True)[0] ) 

        color = (0, 0, 255)
        thickness = 2

        image = cv2.line(image, (red_x[bottom_0], red_y[bottom_0]), (red_x[bottom_1], red_y[bottom_1]), color, thickness)
        image = cv2.line(image, (red_x[upper_0], red_y[upper_0]), (red_x[upper_1], red_y[upper_1]), color, thickness)

        len_bottom = m.sqrt( (red_x[bottom_0]-red_x[bottom_1])**2 + (red_y[bottom_0]-red_y[bottom_1])**2 )
        len_bottom = round( len_bottom * 100/ chopsticks_len , 2)

        len_upper = m.sqrt( (red_x[upper_0]-red_x[upper_1])**2 + (red_y[upper_0]-red_y[upper_1])**2 )
        len_upper = round( len_upper * 100/ chopsticks_len , 2)

        len_diff = round( len_upper-len_bottom, 2)

        pos_x_bottom = int(( red_x[bottom_0]+red_x[bottom_1] ) / 2)
        pos_y_bottom = int(( red_y[bottom_0]+red_y[bottom_1] ) / 2)
        org_bottom = (pos_x_bottom, pos_y_bottom)

        pos_x_upper = int(( red_x[upper_0]+red_x[upper_1] ) / 2)
        pos_y_upper = int(( red_y[upper_0]+red_y[upper_1] ) / 2)
        org_upper = (pos_x_upper, pos_y_upper)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (36,255,12)
        thickness = 1

        image = cv2.putText(image, str(round(len_bottom,2)) + "mm", org_bottom, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
        
        image = cv2.putText(image, str(round(len_upper,2)) + "mm", org_upper, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

        # Get Peak
        if peak_bottom <= len_bottom:
            peak_bottom = len_bottom
        else:
            peak_bottom_get = len_bottom
            peak_bottom = 0
        
        if peak_upper <= len_upper:
            peak_upper = len_upper
        else:
            peak_upper_get = len_upper
            peak_upper = 0

        if peak_diff <= (len_upper-len_bottom):
            peak_diff = (len_upper-len_bottom)
        else:
            peak_diff_get = (len_upper-len_bottom)
            peak_diff = 0

        # Get Angle 
        vec_0 = (abs(red_x[upper_0]-red_x[bottom_0]), abs(red_y[upper_0]-red_y[bottom_0]))
        vec_1 = (abs(red_x[upper_1]-red_x[bottom_1]), abs(red_y[upper_1]-red_y[bottom_1]))    

        unit_vec_0 = vec_0 / np.linalg.norm(vec_0)
        unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
        dot_product = np.dot(unit_vec_0, unit_vec_1)
        
        angle = np.arccos(dot_product)
        angle = round(m.degrees(angle),2)

        pos_x_center = int(( org_bottom[0]+org_upper[0] ) / 2)
        pos_y_center = int(( org_bottom[1]+org_upper[1] ) / 2)
        org_angle = (pos_x_center, pos_y_center)

        str_angle = str(angle) + "deg"
        image = cv2.putText(image, str_angle, org_angle, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    ############################################################### Detect Blue Marker

    blue_lower = np.array([100,150,100])
    blue_upper = np.array([130,255,255])
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    mask = cv2.medianBlur(mask, 15)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    #cv2.imshow('mask', mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Detect ROI countours
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 0

    blue_ROI_number = 0
    blue_x = []
    blue_y = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (255,255,255), 2)
        blue_ROI_number += 1
        blue_x.append(x + int(w/2)) 
        blue_y.append(y + int(h/2))


    if blue_ROI_number == 4:

        # Draw Feature and Line
        blue_order = [ blue_x[0], blue_x[1], blue_x[2], blue_x[3] ]
        
        b_0 = blue_order.index( sorted(blue_order, reverse=True)[3] )
        b_1 = blue_order.index( sorted(blue_order, reverse=True)[2] )
        b_2 = blue_order.index( sorted(blue_order, reverse=True)[1] )
        b_3 = blue_order.index( sorted(blue_order, reverse=True)[0] ) 

        # Draw Feature and Line
        feature_0 = (blue_x[b_0], blue_y[b_0])
        feature_1 = (blue_x[b_1], blue_y[b_1])
        feature_2 = (blue_x[b_2], blue_y[b_2])
        feature_3 = (blue_x[b_3], blue_y[b_3])

        color = (255, 0, 0)
        thickness = 2

        image = cv2.line(image, feature_0, feature_1, color, thickness)
        image = cv2.line(image, feature_1, feature_2, color, thickness)
        image = cv2.line(image, feature_2, feature_3, color, thickness)

        # Get Angle 
        vec_0 = (abs(blue_x[b_0]-blue_x[b_1]), abs(blue_y[b_0]-blue_y[b_1]))
        vec_1 = (abs(blue_x[b_1]-blue_x[b_2]), abs(blue_y[b_1]-blue_y[b_2]))    

        unit_vec_0 = vec_0 / np.linalg.norm(vec_0)
        unit_vec_1 = vec_1 / np.linalg.norm(vec_1)
        dot_product = np.dot(unit_vec_0, unit_vec_1)
        
        angle = np.arccos(dot_product)
        angle = round(m.degrees(angle),2)

        org_angle = (blue_x[b_1], blue_y[b_1] + 50)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (36,255,12)
        thickness = 1

        str_angle = str(180-angle) + "deg"
        image = cv2.putText(image, str_angle, org_angle, font, 
                    fontScale, color, thickness, cv2.LINE_AA)

    # Print Time
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (36,255,12)
    thickness = 2

    org_timer = (25, 50)
    timer = round(time.time()-start_time,2)
    str_time = str(timer) + " second"
    image = cv2.putText(image, str_time, org_timer, font, fontScale, color, thickness, cv2.LINE_AA)

    # Show Image
    cv2.imshow('image', image)
            
    if cv2.waitKey(1) == ord('q'):
        break
    
    # Save To CSV
    if red_ROI_number == 4 and blue_ROI_number == 4:
        writer.writerow([ timer, len_bottom, len_upper,  len_diff, \
                          peak_bottom_get, peak_upper_get, peak_diff_get, (180-angle) ])
        peak_bottom_get = 0 
        peak_upper_get = 0
        peak_diff_get = 0

capture.release()
cv2.destroyAllWindows()
