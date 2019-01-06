"""
Medical Document Cognitive Analysis
"""

import cv2
from PIL import Image
import numpy as np
from keras.preprocessing import image
from tkinter import Tk
from tkinter.filedialog import askopenfilename



# show an "Open" dialog box (window size GUI) and return the path to the selected file
Tk().withdraw()
filename = askopenfilename() 
filename.replace("\t","\\t")


# Reads test image
img1 = cv2.imread (filename, 0)
test_image1 = image.load_img(filename, target_size = (64, 64))

# Executes train_data_module script
#exec(open('train_data_module.py').read())
#import p01

print("-----------")

# Executes test_data script
Var0 = {'test_image':test_image1}
exec(open('test_data.py').read(),Var0)
#import p02

print("-----------")

# Executes image_to_string script
Var1 = {'img':img1}
exec(open('image_to_string.py').read(),Var1)
#import ext