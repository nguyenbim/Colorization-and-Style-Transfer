import requests
import streamlit as st
import base64
import json
import io 
import os
import numpy as np

from PIL import Image
from io import BytesIO

def img_str_to_np_array(img_base64_string):
    img_bytes = base64.b64decode(img_base64_string)
    img_bytesIO = BytesIO(img_bytes)
    img_bytesIO.seek(0)
    image = Image.open(img_bytesIO)
    img_np_arr = np.array(image)
    return img_np_arr

#from utils_sever.strlit_frontend import PROCESS_URL

TESTING_UTILS_LIST = {
        "colorization_model1": "colorization_model1",
        "colorization_model2": "colorization_model2",
        "style_model1": "style_model1",
        "style_model2": "style_model2",
        }

headers = {'content-type': 'application/json'}

st.set_option("deprecation.showfileUploaderEncoding", False)
st.title("Colorization Images and Style Transfer")

image = st.file_uploader("Choose an image")
if image:
    st.image(image, caption = "Your upload image")

test_options = st.selectbox("Choose testing utils", [i for i in TESTING_UTILS_LIST.values()])    

if st.button("Start Process"):
    if image is not None: 
        img_data_bytes = image.getvalue()
        img_base64_bytes = base64.b64encode(img_data_bytes)
        img_base64_str = img_base64_bytes.decode("ascii")


        file_upload = json.dumps({"img_data_str": img_base64_str})
        if test_options:
            res = requests.post(f"http://127.0.0.1:8000/start/{test_options}", file_upload, headers=headers)
            
            try:
                res_dict =  json.loads(res.text)
                img_base64_string = res_dict["img_data_str"]
                result_img =  img_str_to_np_array(img_base64_string)
                # st.json(res_dict)
                result_img = result_img[...,::-1]
                st.image(result_img, caption = "Result image")
            except:
                st.json({"error": "anh khong hop le vui long thu lai anh khac, kiem tra server co bi kill hay khong ??"})


