import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loader(content_img_path, style_img_path, size):
    content_img = Image.open(content_img_path)
    style_img = Image.open(style_img_path)
    transform = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
    content_img_rsz = transform(content_img)
    style_img_rsz = transform(style_img)
    content_img = content_img_rsz.unsqueeze(0)
    style_img = style_img_rsz.unsqueeze(0)

    return content_img.to(device, torch.float), style_img.to(device, torch.float)


def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(-1, h*w)
    G = torch.mm(features, features.t())
    return G.div(b*c*h*w)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

