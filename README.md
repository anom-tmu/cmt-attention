# Chopsticks Manipulation Test with Visual Attention based on Multiscopic Framework for Independent Rehabilitation Support

This application integrate finger measurement and visual attention for Chopsticks manipulation tests (CMT) assessment. We estimate the proximal interphalangeal (PIP) joint angle on the index finger during CMT using fully-connected cascade neural networks (FCC-NN). Then, we implement visual attention measurement during CMT. We propose three parameters, namely joint angle estimation movement (JAEM), chopsticks' attention movement (CAM), and chopsticks' tips movement (CTM), by detecting the local minima and maxima of the signal. In our experimental result, the velocity of these three parameters could indicate improvement in hand and eye function during CMT. This study is expected to benefit hand-eye coordination rehabilitation in the future.

The repository includes:

Source code of 4 steps.
01_feature_extraction.py
02_learning_feature.py
03_testing_model.py
04_data_analysis.py

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). This dataset was created from Kubota Laboratory, Tokyo MEtropolitan University and available for academic use. 

# 1. Microscopic: Feature Extraction

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_multiscopic.jpg)



![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_installation.jpg)



![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_joint_angle_estimation.jpg)



![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_graph.jpg)



![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_eye_tracking.jpg)


