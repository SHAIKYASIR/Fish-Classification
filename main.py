import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import base64
from util import classify, set_background



set_background('images/5076404.jpg')

# set title
st.title('Fish Classification')
st.markdown("""---""")
# set header
st.header('Please upload an image of a fish of the following species:')
st.markdown("""#### 1.Black Sea Sprat 
#### 2.Gilt-Head Bream
#### 3.Hourse Mackerel 
#### 4.Red Mullet
#### 5.Red Sea Bream
#### 6.Sea Bass, Shrimp
#### 7.Striped Red Mullet
#### 8.Shrimp
#### 9.Trout <br>""",True)
st.markdown("""---""")


name= st.text_input("Enter the Fish name to know if Extinct or Not:")

d={
    "Ganges Shark":"Extinct",
    "Pondicherry Shark":"Extinct",
    "Galapagos Shark ":"Extinct",
    "Ganges River Dolphin":"Extinct",
    "Deccan Mahseer":"Extinct",
    "Malabar Mahseer":"Extinct",
    "Humpback Mahseer":"Extinct",
    "Labeo kontius":"Extinct",
    "Garra hughi":"Extinct",
    "Puntius ticto":"Extinct"
}

if name in d:
    st.write("It is", d[name])


ptr=st.write("To know the extinct species press the button")

if st.button("Show"):
    st.write(d)
   
st.markdown("""### Upload Image here """,True)

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
