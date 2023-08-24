import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify, set_background


# set_background('./bgs/bg5.png')

# set title
st.title('Fish Classification')

# set header
st.header('Please upload an image of a fish of the following species: Black Sea Sprat, Gilt-Head Bream, Hourse Mackerel, Red Mullet, Red Sea Bream, Sea Bass, Shrimp, Striped Red Mullet, Trout')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model('model.h5')

# load class names
# with open('./model/labels.txt', 'r') as f:
#     class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
#     f.close()
class_names=['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel', 'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    class_name, conf_score = classify(image, model, class_names)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### score: {}%".format((conf_score*10)))