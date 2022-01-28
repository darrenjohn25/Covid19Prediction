import numpy as np
import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
# new branch changes
st.markdown("<h1 style='text-align: center;'>Squirrel VS Tortoise</h1>", unsafe_allow_html=True)
#change in master
model = tf.keras.models.load_model("resnet_ct.h5")
### load file
file = st.file_uploader("Upload a CT file of Covid 19 in Jpeg ", type="jpg")
classes = {'Normal': 1, 'Covid': 0}
if file is not None:
    image = cv2.imread(file)

    st.image(
        image,
    )
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img_array, (224, 224))
    img = np.array(img / 255.0)
    img = np.expand_dims(img,axis=0)
    prediction = model.predict(img)
    # pred_new= np.argmax(prediction, axis=1)
    st.title("Predicted Label for the image is {}".format(list(classes.keys())[list(classes.values()).index(int(tf.round(prediction)))]))