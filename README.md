# PersonalizedDosimetry
In this project we develop fast computer algorithms to determine a patient-specific safety margin at the start of the MRI examination. This allows to personalize the RF safety constraints on MRI, enabling current and future generations of MRI systems to operate safely at their full potential.

This repository contains a working example code written in Python 3.9 including pre-trained networks, which aims to map T1-weighted images acquired at 7T to a body model represented in 8 distinct tissue classes. The tissue labels are defined as follows: Background (0), internal air (1), bone (2), muscle (3), fat (4), WM (5), GM (6), CSF (7), eye (8). The code is compatible with input and output datasets in NIfTI format. Details on the background of the method as well as MR acquisition sequence can be found in doi:T.B.A. .
