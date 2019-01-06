"""
 OBJECTIVE:
    To Classify image based on their hospital by loading the trained data module and
    predicting the hospital using keras
"""
# Importing the Keras libraries and packages
import numpy as np
from keras.preprocessing import image
from PIL import Image
from keras.models import Sequential
from keras.models import load_model
import cv2 


# Loading trained data module
model = load_model('trained_data_model')

# Loading test image
#test_image = image.load_img(r'C:\Users\aditya\Desktop\yup\ac2.jpg', target_size = (64, 64))

# Converting image to array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# Find the test_image hospital
result = model.predict_classes(test_image)


# Printing test_image class
if result[0][0] == 0:
    prediction = 'global'
else :
	prediction = 'apollo'

print(prediction)

	