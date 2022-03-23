# Chopsticks Manipulation Test with Visual Attention based on Multiscopic Framework for Independent Rehabilitation Support

This application integrate finger measurement and visual attention for Chopsticks manipulation tests (CMT) assessment. We estimate the proximal interphalangeal (PIP) joint angle on the index finger during CMT using fully-connected cascade neural networks (FCC-NN). Then, we implement visual attention measurement during CMT. We propose three parameters, namely joint angle estimation movement (JAEM), chopsticks' attention movement (CAM), and chopsticks' tips movement (CTM), by detecting the local minima and maxima of the signal. In our experimental result, the velocity of these three parameters could indicate improvement in hand and eye function during CMT. This study is expected to benefit hand-eye coordination rehabilitation in the future.

The repository includes:
. Data training for joint angle estimation
. Data testing for Chopsticks Manipulation Test (CMT)
. Source code (01_feature_extraction.py, 02_learning_feature.py, 03_testing_model.py, 04_data_analysis.py).
. Pre-trained model and weights

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). This dataset was created from Kubota Laboratory, Tokyo Metropolitan University and available for academic use. 


This figure shows the design of multiscopic cyber-physical-social systems for chopsticks manipulation test. This approach includes a) modeling finger kinematic and chopsticks state for feature extraction at the microscopic level, b) estimating PIP finger feature  [37] by active perception ability at the mesoscopic level, and (c) discovering the human visual attention by cognition ability at the macroscopic levels. 

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_multiscopic.jpg)

We explain the microscopic, mesoscopic, and macroscopic levels in three sub-sections. We describe our method from setting up the experiments, extracting finger features, detecting chopsticks state, estimating the joint angle, and measuring visual attention during the CMT.


## 1. Microscopic Level: Feature Extraction

We used two markers: chopsticks markers in red and finger markers represented in blue. We performed marker detection on the index finger because it was visible in egocentric vision, while the middle finger was closed. We got the finger features by connecting the center of the blue marker with the center of the nearest blue marker to form a skeletal finger. We used smart glasses as non-contact wearable sensors on egocentric view and set them up in the direction of participant view. 

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_installation.jpg)


## 2. Mesoscopic Level: Active Perception

We used Cheng-type FCC-NN [45] and compared it with the conventional architecture such as multi-layer perceptron (MLP). We applied the upper length (L_up) and bottom length (L_bot) as input of neural networks to get PIP joint angle (θ) as output. We collected data for 1 minute for all chopsticks manipulation poses with different directions. 

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_joint_angle_estimation.jpg)

We have successfully implemented FCC-NN as a learning algorithm to estimate PIP joint angle in the preliminary test. The upper length (L_up) and bottom length (L_bot) as inputs for the networks and the PIP joint angle (θ) as the output. We compare this result with MLP with several hidden nodes and evaluate the performance. This figure shows PIP joint angle estimation training and testing results using MLP and FCC-NN. 

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_graph.jpg)

## 3. Macroscopic Level: Cognitive Ability

The direction of attention usually is focused on the object and the tip of the chopsticks. The movement of the chopstick tip also follows the person's attention. We utilized Tobii Pro Glasses 3 to get attention on the image plane [48]. This intelligent eyeglass gives us information about the attention location from a red circle in the image area. The location of the center of attention can fastly change according to the movement of the eyes. We also put blue markers on both tips of the chopsticks to determine the location of the chopstick tips in the image coordinate. 

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_eye_tracking.jpg)


