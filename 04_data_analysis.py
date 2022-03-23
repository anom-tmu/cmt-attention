import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Prepare the Figure
figure, ax = plt.subplots(figsize=(10, 3))
figure, ay = plt.subplots(figsize=(10, 3))
figure, az = plt.subplots()
figure, aq = plt.subplots()

# Read the CSV data
df = pd.read_csv('C:\User_Folder\cmt_result.csv')

timer = df['timer']
joint = df['joint_angle']
chops = -1 * df['mean_blue_y']
eye = -1 * df['eye_y']



###### Chopsticks Attention Movement (CAM)

diff_eye = [0]
for i in range(0, len(eye)-1):
    diff_eye.append(eye.values[i] - eye.values[i+1])
diff_eye = np.array(diff_eye)

# Peaks for eye_signal
#peaks_0,_ = find_peaks(eye, distance = 1, prominence=50) 

# Peaks for timer_eye
peaks_2,_ = find_peaks(diff_eye, distance = 1, prominence=50)
peaks_3,_ = find_peaks( (-1*diff_eye) , distance = 1, prominence=50) 

# Count the differential of signals
timer_eye = []
distance_eye = []
speed_eye = []

for i in range (len(peaks_3)-1):
    for j in range (len(peaks_2)-1):
        
        timer_temp1 = round( timer[peaks_2[j]] - timer[peaks_3[i]], 3) 
        distance_temp1 = round( abs( eye[peaks_2[j]] - eye[peaks_3[i]]), 0)
        speed_temp1 = round (distance_temp1 / (timer_temp1 * 1000), 3)
        
        if timer_temp1 > 0.1 and timer_temp1 <0.5:
            timer_eye.append(timer_temp1 * 1000) 
            distance_eye.append(distance_temp1)
            speed_eye.append(speed_temp1)

timer_eye_avg = round( sum(timer_eye) / len(timer_eye), 2)
print('CAM Mean Time  : ' + str(timer_eye_avg) + ' ms')

distance_eye_avg = round( sum(distance_eye) / len(distance_eye), 2)
print('CAM Mean Dist  : ' + str(distance_eye_avg) + ' pixel')

speed_eye_avg = round( sum(speed_eye) / len(speed_eye), 3)
print('CAM Mean Speed : ' + str(speed_eye_avg) + ' pixel/ms')

ax.plot(timer,  -1 * diff_eye, color="pink")
ax.plot(timer[peaks_2], -1 * diff_eye[peaks_2], "o", color="blue")
ax.plot(timer[peaks_3], -1 * diff_eye[peaks_3], "o", color="red")
ax.set_ylim(-150, 150)
ax.set_xlim(0, 30)

ax.set_xlabel("time (second)")
ax.set_ylabel("distance (pixel)")
ax.set_title("Chopsticks Attention Movements (CAM)")



###### Chopsticks Tips Movement (CTM)

diff_chops = [0]
for i in range(0, len(chops)-1):
    diff_chops.append(chops.values[i] - chops.values[i+1])
diff_chops = np.array(diff_chops)

# Peaks for chopsticks blue marker
# peaks_1,_ = find_peaks(chops, distance = 1, prominence=50) 

# Peaks for timer_chops
peaks_4,_ = find_peaks(diff_chops, distance = 1, prominence=50)
peaks_5,_ = find_peaks( (-1*diff_chops) , distance = 1, prominence=50) 

# Count the differential of signals
timer_chops = []
distance_chops = []
speed_chops = []

for i in range (len(peaks_5)-1):
    for j in range (len(peaks_4)-1):
        
        timer_temp2 = round( timer[peaks_4[j]] - timer[peaks_5[i]], 3) 
        distance_temp2 = round( abs( chops[peaks_4[j]] - chops[peaks_5[i]]), 0)
        speed_temp2 = round (distance_temp2 / (timer_temp2 * 1000), 3)
        
        if timer_temp2 > 0.1 and timer_temp2 <0.5:
            timer_chops.append(timer_temp2 * 1000) 
            distance_chops.append(distance_temp2)
            speed_chops.append(speed_temp2)

print('---')
timer_chops_avg = round( sum(timer_chops) / len(timer_chops), 2)
print('CTM Mean Time  : ' + str(timer_chops_avg) + ' ms')

distance_chops_avg = round( sum(distance_chops) / len(distance_chops), 2)
print('CTM Mean Dist  : ' + str(distance_chops_avg) + ' pixel')

speed_chops_avg = round( sum(speed_chops) / len(speed_chops), 3)
print('CTM Mean Speed : ' + str(speed_chops_avg) + ' pixel/ms')

ay.plot(timer, -1 * diff_chops, color="lightblue")
ay.plot(timer[peaks_4], -1 * diff_chops[peaks_4], "o", color="blue")
ay.plot(timer[peaks_5], -1 * diff_chops[peaks_5], "o", color="red")
ay.set_ylim(-150, 150)
ay.set_xlim(0, 30)

ay.set_xlabel("time (second)")
ay.set_ylabel("distance (pixel)")
ay.set_title("Chopsticks Tips Movements (CTM)")



###### Joint Angle Estimation Movement (JAEM)

az.plot(timer, joint, color="lightgrey")
joint2 = savgol_filter(joint, 5, 3) #35
diff_joint = joint2

az.plot(timer, joint2, color="blue")

az.set_xlim(0, 30)
az.set_ylim(145, 165)
az.set_xlabel("time (second)")
az.set_ylabel("angle (degree)")
az.set_title("Original PIP joint angle estimation")

peaks_7, _ = find_peaks(joint2, distance = 1, prominence=3)
peaks_6, _ = find_peaks(-1*joint2, distance = 1, prominence=3) 

# Count the differential of signals
timer_joint = []
distance_joint = []
speed_joint = []

for i in range (len(peaks_7)-1):
    for j in range (len(peaks_6)-1):
        
        timer_temp3 = round( timer[peaks_6[j]] - timer[peaks_7[i]], 3) 
        distance_temp3 = round( abs( joint[peaks_6[j]] - joint[peaks_7[i]]), 0)
        speed_temp3 = round (distance_temp3 / (timer_temp3 * 1000), 3)
        
        if timer_temp3 > 0.1 and timer_temp3 <0.5:
            timer_joint.append(timer_temp3 * 1000) 
            distance_joint.append(distance_temp3)
            speed_joint.append(speed_temp3)

print('---')
timer_joint_avg = round( sum(timer_joint) / len(timer_joint), 2)
print('JAEM Mean Time : ' + str(timer_joint_avg) + ' ms')

distance_joint_avg = round( sum(distance_joint) / len(distance_joint), 2)
print('JAEM Mean Dist : ' + str(distance_joint_avg) + ' degree')

speed_joint_avg = round( sum(speed_joint) / len(speed_joint), 3)
print('JAEM Mean Speed: ' + str(speed_joint_avg) + ' degree/ms')

aq.plot(timer, joint2, color="orange")
aq.plot(timer[peaks_7], joint2[peaks_7], "o", color="red")
aq.plot(timer[peaks_6], joint2[peaks_6], "o", color="blue")

aq.set_xlim(0, 30)
aq.set_ylim(145, 165)
aq.set_xlabel("time (second)")
aq.set_ylabel("angle (degree)")
aq.set_title("Peak and valley of PIP joint angle estimation")

plt.show()