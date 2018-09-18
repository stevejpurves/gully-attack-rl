from os import path
import numpy as np
from scipy import misc

DATA_PATH=path.join('..','..','Parihaka')

LABEL_PATH=path.join(DATA_PATH, 'labels')
IMAGE_PATH=path.join(DATA_PATH, 'images')

inlines = [2300]

seismic_names = ['grey_il{}_segmentation.png'.format(il) for il in inlines]
label_names = ['rgb_il{}_segmentation.png'.format(il) for il in inlines]

for s_name, l_name in zip(seismic_names, label_names):
  image = misc.imread(l_name)
  print(image.shape)

