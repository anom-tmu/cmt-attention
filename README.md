# Chopsticks Manipulation Test with Visual Attention based on Multiscopic Framework for Independent Rehabilitation Support


![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/cover_cmt.gif)


This research integrate finger measurement and visual attention for chopsticks manipulation tests (CMT) assessment. We estimate the proximal interphalangeal (PIP) joint angle on the index finger during CMT using fully-connected cascade neural networks (FCC-NN). Then, we implement visual attention measurement using Tobii Pro Glasses 3 during CMT.

![alt text](https://github.com/anom-tmu/cmt-attention/blob/main/pic_installation.jpg)

By detecting the local minima and maxima of the signal, we propose three parameters, namely joint angle estimation movement (JAEM), chopsticks' attention movement (CAM), and chopsticks' tips movement (CTM). In our experimental result, the velocity of these parameters could indicate improvement in hand and eye function during CMT. This study is expected to benefit hand-eye coordination rehabilitation in the future.


## Contents
The repository includes:
  - Data training for joint angle estimation
  - Data testing for Chopsticks Manipulation Test (CMT)
  - Pre-trained model and weights
  - Source code:
    - 01_feature_extraction.py
    - 02_learning_feature.py
    - 03_testing_model.py
    - 04_data_analysis.py

The code is documented and designed to be easy to extend. If you use it in your research, please consider citing this repository (bibtex below). This dataset was created from Kubota Laboratory, Tokyo Metropolitan University and available for academic use. 

## Requirements
This system works with this packages:
  - python 3.8.5
  - pytorch 1.9.0
  - opencv-python 4.5.2.54 
  - numpy 1.19.1
  - pandas 1.1.3
  - scipy 1.5.2
  - other common packages listed in source code.

## Contributing
Contributions to this repository are welcome. Examples of things you can contribute:
  - Speed Improvements.
  - Training on other datasets.
  - Accuracy Improvements.
  - Visualizations and examples.
  - Join our team and help us build even more projects.

## Citation
Use this bibtex to cite this repository: 
```
This research not yet published
```
