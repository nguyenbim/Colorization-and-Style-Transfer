
import torch
import torch.nn as nn
import numpy as np
from IPython import embed
import cv2

from src1.util_colorization import *
from src1.eccv16 import ECCVGenerator


class Colorization(ECCVGenerator):
    def __init__(self, path_model, device):
        super(Colorization, self).__init__()
        self.load_state_dict(torch.load(path_model))
        self.eval()
        self.device = device

    def predict(self, img):
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        model_input = tens_l_rs.to(self.device)
        predicted_ab = self(model_input)

        image_rgb = postprocess_tens(tens_l_orig, predicted_ab)

        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        return image_rgb