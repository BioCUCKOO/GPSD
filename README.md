# GPSD
**Note**: The GPSD, which integrated 10 sequence features and used 2 types of machine learning, including penalized logistic regression (PLR) and deep neural networks (DNN), was developed to predict novel dephospharylation sites. 

## Requirements

The main requirements are listed below:

* Python 3.8
* Scikit-Learn
* Joblib
* Keras
* Numpy
* Pandas


## The description of GPSD source codes

* GPSD_prediction.py

    The code is used for the score and input DSP(30,30). The closer the score is to 1, the more likely it is that this site is a dephosphorylation site.


## OS Requirements

Above codes have been tested on the following systems:

* Windows: Windos10

## Hardware Requirements

All codes and softwares could run on a "normal" desktop computer, no non-standard hardware is needed.

## Installation guide

All codes can run directly on a "normal" computer with Python 3.8 installed, no extra installation is required.

## Instruction

For users who want to run GPSD in own computer, you should import the required packages. Then set the correct file path, including input and output files, and finally run the GPSD_prediction.py file of the model to obtain the result file.

