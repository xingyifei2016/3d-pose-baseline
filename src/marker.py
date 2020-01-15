
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cameras
import viz
import h5py
import glob
import copy
import procrustes

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1,5,6,7,8]
TEST_SUBJECTS  = [9,11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'

def markup_36m():
  img = list(open("/mnt/lustre/xingyifei/to_ptx/train_images.txt", "r"))[0][:-1]
  params = h5py.File('/mnt/lustre/xingyifei/to_ptx/train.h5', 'r')['GT2d'][0].astype(int)
  img_open = plt.imread('/mnt/lustre/xingyifei/to_ptx/images/'+img)
  img_open = img_open.copy()

  for i in params:
    img_open[i[0]][i[1]] = [255, 255, 0]
    for j in [-1, 1]:
      for k in [-1, 1]:
        img_open[i[0]+j][i[1]+k] = [255, 255, 0]

  plt.imshow(img_open)
  plt.savefig("Sample1.png")
  return

def markup_3dhp():
  img = list(open("/mnt/lustre/xingyifei/test_3dhp/test_sub_images.txt", "r"))[0][:-1]
  params = h5py.File('/mnt/lustre/xingyifei/test_3dhp/annotTest.h5', 'r')['annot_2d'][0].astype(int)
  img_open = plt.imread('/mnt/lustre/xingyifei/test_3dhp/test_images_full/'+img)
  img_open = img_open.copy()

  for i in params:
    img_open[i[0]][i[1]] = [255, 255, 0]
    for j in [-5, -4, -3, -2,-1, 1,2, 3, 4, 5]:
      for k in [-5, -4, -3, -2,-1, 1,2, 3, 4, 5]:
        img_open[i[0]+j][i[1]+k] = [255, 0, 0]

  plt.imshow(img_open)
  plt.savefig("Sample1.png")
  return

markup_3dhp()









