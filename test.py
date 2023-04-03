import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2
import scipy.io as sio


trimap = Image.open('Image_Source\Trimap\gandalf-trimap.png')
trimap = np.array(trimap).astype(np.float64) 

data = {'trimap': trimap}

# Save the dictionary to a .mat file
sio.savemat('trimap.mat', data)
