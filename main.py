import cv2
import glob
import sys
import numpy as np

sys.path.append("src2")
from deoldify.visualize import *
from pathlib import Path
from src1.colorization import Colorization


render_factor=20 # img_size = render_factor * 16 = 16*20 = 320
colorizer = get_image_colorizer(root_folder=Path("model"),render_factor=render_factor, artistic=True)
print(colorizer.filter.filters[0].learn.model)
list_img = glob.glob('data/*.jpg')
np.random.shuffle(list_img)
exit(0)

def predict_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), 30, size)

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            img =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = colorizer.plot_instance_image(
                img,
                render_factor=render_factor
            )
            result.write(frame)
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()
# predict_video("/home/duong/Downloads/colorization/shortcut_anh_ho.mp4","predict.mp4")
colorizer= VideoColorizer(colorizer)
colorizer._colorize_from_path(Path("/home/duong/Downloads/colorization/shortcut_anh_ho.mp4"))
