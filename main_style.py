import cv2
import matplotlib.pyplot as plt

from PIL import Image
from style.utility import *
from torch.autograd import Variable
from style.Defining_Models import TransformerNet
from torchvision.utils import save_image


path_model = 'model/best_style_tsunami.pth'
# path_model = 'model/best_model_style_2.pth'
image_path = 'data/test/2_256_flower.jpg'
path_save = 'result_images/output.png'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = test_transform()

# Load
model = TransformerNet()
model.load_state_dict(torch.load(path_model, map_location=device))

img = Image.open(image_path)
# Prepare input
image_tensor = Variable(transform(img)).to(device)
image_tensor = image_tensor.unsqueeze(0)

# Stylize image
with torch.no_grad():
    stylized_image = denormalize(model(image_tensor)).cpu()

# Save image
save_image(stylized_image, path_save)


#show image
Image.open(path_save).show()