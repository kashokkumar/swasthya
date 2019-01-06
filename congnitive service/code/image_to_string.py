"""
 OBJECTIVE:
    To Extract details from image by using OCR and PYTESSERACT to convert image to string
    and by using regular expressions with nltk(natural language tool kit) to get details
"""

# Importing the pytesseract libraries ,nltk libraries and their packages
import pytesseract
from PIL import Image
import numpy as np
import cv2
import re
import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')


# Read image using opencv
#img = cv2.imread (r"C:\Users\aditya\Desktop\check\hi1.jpg", 0)

# Rescaling the image.
img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

# Apply dilation and erosion to remove some noise
kernel = np.ones((1, 1), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN,kernel)

# Apply threshold to get image with only b&w (binarization)
img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Recognize text with tesseract for python
result = pytesseract.image_to_string(img, lang="eng")
print (result)

# Using function to extracts phone number from the text 
def extract_phone_numbers(result):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(result)
    return [re.sub(r'\D', '', number) for number in phone_numbers]

# Using function to extracts email id from the text    
def extract_email_addresses(result):
    r = re.compile(r'\S+@\S+')
    return r.findall(result)

# Using function to word tokinizes from the text
def ie_preprocess(document):
    document = ' '.join([i for i in document.split() if i not in stop])
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences

# Using function to extracts names from the list
def extract_names(document):
    names = []
    sentences = ie_preprocess(document)
    for tagged_sentence in sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if type(chunk) == nltk.tree.Tree:
                if chunk.label() == 'PERSON':
                    names.append(' '.join([c[0] for c in chunk]))
    return names


#output from the functions are displayed
numbers = extract_phone_numbers(result)
emails = extract_email_addresses(result)
names = extract_names(result)
print("\nphone no.s:-")
print(numbers)
print("\nemails:-")
print(emails)
print("\nnames:-")
print(names)