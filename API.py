import base64
import sys
import requests, json
import os
import uvicorn
import cv2
import numpy as np 
import uvicorn
import matplotlib.pyplot as plt

from logbook import Logger
from fastapi import Depends, HTTPException, Request, APIRouter, BackgroundTasks
from fastapi import responses, status
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from jsonschema import ValidationError
from io import BytesIO
from PIL import Image, ImageDraw
from pathlib import Path
import streamlit as st
import torch

from src1.colorization import Colorization
sys.path.append("src2")
from class_object import ImagesFromCLients
from deoldify.visualize import *
from style.utility import *
from torch.autograd import Variable
from style.Defining_Models import TransformerNet
from torchvision.utils import save_image


logger = Logger(__name__)
router = APIRouter()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## colorization_model1_function
path_colorization_model1 = 'model/model_colorization.pth'
colorization_model1 = Colorization(path_colorization_model1, device='cpu')

## colorization_model2_function
render_factor=20 # img_size = render_factor * 16 = 16*20 = 320
colorizer = get_image_colorizer(root_folder=Path("model"),render_factor=render_factor, artistic=True)

def predict_image(cap, model):
    cap = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    predict = model.vis.plot_instance_image(cap, render_factor=20)
    return predict
colorizer= VideoColorizer(colorizer)

## style_model1_function
def style_predict(img, model):
    image_tensor = Variable(transform(img)).to(device)
    image_tensor = image_tensor.unsqueeze(0)

    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(model(image_tensor)).cpu()

    # Save image
    path_save = 'resource_images/output_style.png'
    save_image(stylized_image, path_save)

    #show image
    return cv2.imread(path_save)

path_style_model1 = 'model/best_style_tsunami.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = test_transform()
# Load
style_model1 = TransformerNet()
style_model1.load_state_dict(torch.load(path_style_model1, map_location=device))


## style_model1_functio

path_style_model2 = 'model/best_model_style_2.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = test_transform()
# Load
style_model2 = TransformerNet()
style_model2.load_state_dict(torch.load(path_style_model2, map_location=device))


def img_str_to_np_array(img_base64_string):
    img_bytes = base64.b64decode(img_base64_string)
    img_bytesIO = BytesIO(img_bytes)
    img_bytesIO.seek(0)
    image = Image.open(img_bytesIO)
    img_np_arr = np.array(image)
    return img_np_arr


def img_to_base64(img_result):
    img_result = Image.fromarray(img_result)
    img_result_bytes = BytesIO()
    img_result.save(img_result_bytes, format="PNG")
    img_result_bytes = img_result_bytes.getvalue()
    img_result_base64_bytes = base64.b64encode(img_result_bytes)
    img_result_base64_str = img_result_base64_bytes.decode("ascii")
    return img_result_base64_str


@router.post("/start/colorization_model1/", response_model=ImagesFromCLients)
def colorization_model1_function(img_upload : ImagesFromCLients):
    img_np_arr = img_str_to_np_array(img_upload.img_data_str)

    img_result = colorization_model1.predict(img_np_arr)
    img_result = np.uint8(255 * img_result)

    img_result_base64_str = img_to_base64(img_result)
    return {"img_data_str": img_result_base64_str}


@router.post("/start/colorization_model2/", response_model=ImagesFromCLients)
def colorization_model2_function(img_upload : ImagesFromCLients):
    img_np_arr = img_str_to_np_array(img_upload.img_data_str)

    img_result = predict_image(img_np_arr,colorizer)

    img_result_base64_str = img_to_base64(img_result)
    return {"img_data_str": img_result_base64_str}


@router.post("/start/style_model1/", response_model=ImagesFromCLients)
def style_model1_function(img_upload : ImagesFromCLients):
    img_np_arr = img_str_to_np_array(img_upload.img_data_str)

    img_result = style_predict(img_np_arr, style_model1)

    img_result_base64_str = img_to_base64(img_result)
    return {"img_data_str": img_result_base64_str}


@router.post("/start/style_model2/", response_model=ImagesFromCLients)
def style_model2_function(img_upload : ImagesFromCLients):
    img_np_arr = img_str_to_np_array(img_upload.img_data_str)

    img_result = style_predict(img_np_arr, style_model2)

    img_result_base64_str = img_to_base64(img_result)
    return {"img_data_str": img_result_base64_str}