# Import the necessary packages
import numpy as np
import math as m
import cv2
import torch
import torch.nn as nn
import time
import csv
import pickle
from statistics import mean
import matplotlib.pyplot as plt

# Create CSV file
header = ['timer', 'len_bottom', 'len_upper', 'len_diff', 'peak_bottom', 'peak_upper', 'peak_diff', 'joint_angle', 'len_blue', 'blue1_eye', 'blue2_eye', 'eye_x', 'eye_y', 'mean_blue_x', 'mean_blue_y'] 
        # 'angle0', 'angle1', 'angle2'
csvfile = open('chopsticks01.csv', 'w')
writer = csv.writer(csvfile, delimiter = ',', lineterminator='\n')
writer.writerow(header)

start_time = time.time()

# Fully Connected Neural Networks with One Hidden Layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class Cascade(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Cascade, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(input_size + hidden_size, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2( torch.cat((x,out), dim = 1)  )
        return out

# Device configuration - Open this line when you use CUDA GPU 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters - Choose one model (MLP or Cascade NN) 
# MLP
#input_size = 2
#hidden_size = 3
#output_size = 1

# Cascade
input_size = 2
hidden_size = 1
output_size = 1

# Defining ANN Architechture
#model = MLP(input_size, hidden_size, output_size)         #.to(device)
model = Cascade(input_size, hidden_size, output_size)      #.to(device)

model.load_state_dict(torch.load(r"C:\User_Folder\model.pkl"))
#model.to(device)
model.eval()

sc_input = pickle.load(open(r"C:\User_Folder\scaler_input.pkl",'rb'))
sc_output = pickle.load(open(r"C:\User_Folder\scaler_output.pkl",'rb'))

# Load Video

PATH = r"C:\User_Folder\cmt_testing.mp4"
capture = cv2.VideoCapture(PATH)

if not capture.isOpened():
    print("Cannot open file/camera")
    exit()

# Initialization
peak_bottom = 0
peak_upper = 0
peak_diff = 0
peak_bottom_get = 0
peak_upper_get = 0
peak_diff_get = 0
timer = 0

eye_x = 0
eye_y = 0

len_blue = 0
len_blue1_eye = 0
len_blue2_eye = 0

list_eye_x = []
list_eye_y = [] 

while timer < 60:
    
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
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    # Copy and Convert RGB to HSV
    image = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    ############################################################### Detect Eye-Attention Spot
    
    eye_lower = np.array([0,175,125])
    eye_upper = np.array([5,255,255])
    mask_eye = cv2.inRange(hsv, eye_lower , eye_upper)

    kernel = np.ones((3,3), np.uint8)
    mask_eye = cv2.dilate(mask_eye, kernel, iterations=20)
    #cv2.imshow('mask_eye', mask_eye)

    # Detect ROI countours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(mask_eye, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 0

    eye_ROI_number = 0

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (255,255,255), 3)
        eye_ROI_number += 1
        eye_x = x + int(w/2)
        eye_y = y + int(h/2)

    cv2.circle(image, (eye_x, eye_y), radius=5, color=(0, 0, 255), thickness=-1) #(36,255,12)
    
    list_eye_x.append(eye_x)
    list_eye_y.append(eye_y)
    
    # Brigtness and Contrast
    alpha = 2
    beta = 5
    image = cv2.addWeighted(image, alpha, np.zeros(image.shape, image.dtype), 0, beta)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    ############################################################### Detect Red Marker (Chopsticks Handle)

    red_lower = np.array([0,100,50])
    red_upper = np.array([5,255,150]) 

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    
    red_mask = cv2.medianBlur(red_mask,5)
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.dilate(red_mask, kernel, iterations=3)
    #cv2.imshow('red_mask', red_mask)
 
     # Detect ROI countours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 0

    red_ROI_number = 0
    red_x = []
    red_y = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (0,0,255), 2)
        red_ROI_number += 1
        red_x.append(x + int(w/2)) 
        red_y.append(y + int(h/2))

    if red_ROI_number == 4:

        # Draw Feature and Line
        feature_0 = (red_x[0], red_y[0])
        feature_1 = (red_x[1], red_y[1])
        feature_2 = (red_x[2], red_y[2])
        feature_3 = (red_x[3], red_y[3])

        color = (36,255,12) #(0, 0, 255)
        thickness = 1

        image = cv2.line(image, (red_x[0], red_y[0]), (red_x[1], red_y[1]), color, thickness)
        image = cv2.line(image, (red_x[0], red_y[0]), (red_x[2], red_y[2]), color, thickness)
        image = cv2.line(image, (red_x[0], red_y[0]), (red_x[3], red_y[3]), color, thickness)
        image = cv2.line(image, (red_x[1], red_y[1]), (red_x[2], red_y[2]), color, thickness)
        image = cv2.line(image, (red_x[1], red_y[1]), (red_x[3], red_y[3]), color, thickness)
        image = cv2.line(image, (red_x[2], red_y[2]), (red_x[3], red_y[3]), color, thickness)

        # Get Chopsticks Len
        len_0 = m.sqrt( (red_x[0]-red_x[1])**2 + (red_y[0]-red_y[1])**2 )
        len_1 = m.sqrt( (red_x[1]-red_x[2])**2 + (red_y[1]-red_y[2])**2 )
        len_2 = m.sqrt( (red_x[2]-red_x[3])**2 + (red_y[2]-red_y[3])**2 )
        len_3 = m.sqrt( (red_x[3]-red_x[0])**2 + (red_y[3]-red_y[0])**2 )
        len_4 = m.sqrt( (red_x[0]-red_x[2])**2 + (red_y[0]-red_y[2])**2 )
        len_5 = m.sqrt( (red_x[1]-red_x[3])**2 + (red_y[1]-red_y[3])**2 )

        len_order = sorted([ len_0, len_1, len_2, len_3, len_4, len_5 ])

        chopsticks_len = 0
        chopsticks_len = sorted(len_order, reverse=True)[2]

        len_bottom = len_order[0]
        len_bottom = round( len_bottom * 100/ chopsticks_len , 2)

        len_upper = len_order[1]
        len_upper = round( len_upper * 100/ chopsticks_len , 2)

        len_diff = round( len_upper-len_bottom, 2)


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


    ############################################################### Detect Blue Marker (Chopsticks Tips)
    
    blue_lower = np.array([100,100,0])
    blue_upper = np.array([150,255,255])
    
    # ROI Blue
    blue_roi = 50
    blue_mask = np.zeros((height, width), np.uint8)
    blue_mask[eye_y-blue_roi:+eye_y+blue_roi, eye_x-blue_roi:+eye_x+blue_roi] = cv2.inRange(hsv[eye_y-blue_roi:+eye_y+blue_roi, eye_x-blue_roi:+eye_x+blue_roi], blue_lower, blue_upper)

    blue_mask = cv2.medianBlur(blue_mask, 3)
    kernel = np.ones((3,3), np.uint8)
    blue_mask = cv2.dilate(blue_mask, kernel, iterations=2) #3   
    #cv2.imshow('blue_mask', blue_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    offset = 0

    blue_ROI_number = 0
    blue_x = []
    blue_y = []

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x - offset, y - offset), (x + w + offset, y + h + offset), (255,0,0), 2) #
        blue_ROI_number += 1
        blue_x.append(x + int(w/2)) 
        blue_y.append(y + int(h/2))
    
    color = (36,255,12)
    thickness = 2
    
    
    if blue_ROI_number == 1 : 
        image = cv2.line(image, (blue_x[0],blue_y[0]) , (eye_x, eye_y), color, thickness)

    elif blue_ROI_number == 2 :
        # Draw Line
        image = cv2.line(image, (blue_x[0],blue_y[0]) , (blue_x[1],blue_y[1]), color, thickness)
        image = cv2.line(image, (blue_x[0],blue_y[0]) , (eye_x, eye_y), color, thickness)
        image = cv2.line(image, (blue_x[1],blue_y[1]) , (eye_x, eye_y), color, thickness)

        # Get Distance
        len_blue = m.sqrt( (blue_x[0]-blue_x[1])**2 + (blue_y[0]-blue_y[1])**2 )
        len_blue1_eye = m.sqrt( (blue_x[0]-eye_x)**2 + (blue_y[0]-eye_y)**2 )
        len_blue2_eye = m.sqrt( (blue_x[1]-eye_x)**2 + (blue_y[1]-eye_y)**2 )

        len_blue = round( len_blue , 2)    
        len_blue1_eye = round( len_blue1_eye , 2)
        len_blue2_eye = round( len_blue2_eye , 2)
        #print(len_blue, len_blue1_eye, len_blue2_eye)


    # Print Time
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (36,255,12)
    thickness = 1

    org_timer = (25, 50)
    timer = round(time.time()-start_time,2)
    str_time = "time: " + str(timer) + " sec"
                
    if cv2.waitKey(1) == ord('q'):
        break
    


    # Estimate PIP Joint Angle
    
    if (len_bottom!=0) and (len_upper!=0):
        
        x_data = [[len_bottom, len_upper]]
        x_data = np.array(x_data)

        x_train = sc_input.transform(x_data)
        x_train = torch.tensor(x_train, dtype=torch.float32)
        
        predict = model(x_train)
        np_predict = predict.to('cpu').detach().numpy().copy()
        sc_predict = sc_output.inverse_transform(np_predict.tolist(), None)
        angle_estimate = round(sc_predict[0][0],2)
        #angle_error = round( abs(angle_estimate - (180-angle))/ (180-angle), 3)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (36,255,12)
        thickness = 1
        
        str_len_bottom = "len_bottom : " + str(round(len_bottom,2)) + " mm"
        str_len_upper  =  "len_upper  : " + str(round(len_upper,2)) + " mm"
        str_angle_estimate = "joint est. : " + str(angle_estimate) + " deg" # + ", err: " + str(angle_error)
        attention = round(  max(len_blue1_eye, len_blue2_eye) * 100/ chopsticks_len , 2)
        str_attention = "attention  : " + str(attention) + " mm"
        
        image = cv2.putText(image, str_len_bottom , (25, 25), font, fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, str_len_upper, (25, 50), font, fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, str_angle_estimate, (25, 75), font, fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, str_attention, (25, 100), font, fontScale, color, thickness, cv2.LINE_AA)


    # Save to CSV
    if blue_ROI_number == 2 and red_ROI_number == 4:
        writer.writerow([ timer, len_bottom, len_upper,  len_diff, \
                          peak_bottom_get, peak_upper_get, peak_diff_get, angle_estimate, len_blue, len_blue1_eye, len_blue2_eye, eye_x, eye_y, mean(blue_x), mean(blue_y) ] ) #(180-angle)
        peak_bottom_get = 0 
        peak_upper_get = 0
        peak_diff_get = 0

    cv2.imshow('image', image)


# Plot Heatmap

import matplotlib.cm as cm
from scipy.ndimage.filters import gaussian_filter

def myplot(x, y, s, bins=1000):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

fig, ax = plt.subplots()

x = list_eye_x
y = list_eye_y
s = 16 #8, 16, 32, 64

img, extent = myplot(x, y, s)
ax.imshow(img, extent=extent, origin="upper", cmap=cm.jet)  # origin="lower"
ax.set_title("Smoothing with  $\sigma$ = %d" % s)

plt.show()