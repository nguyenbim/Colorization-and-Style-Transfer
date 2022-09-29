import cv2
import glob
import numpy as np

from src1.colorization import Colorization

def predict_video(video_path, save_path,model):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(save_path,cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = model.predict(frame)
            frame = cv2.resize(frame, (frame_width,frame_height))
            frame = np.uint8(255 * frame)
            result.write(frame)
        else:
            break
    cap.release()
    result.release()
    cv2.destroyAllWindows()

path_model = 'model/model_colorization.pth'
model = Colorization(path_model, device='cpu')

path_img = 'data/test/391_256_bird.jpg'

predict_image =  True

if predict_image:
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = model.predict(img)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    video_path = 'video_cut.mp4'
    save_path = 'predict.mp4'
    predict_video(video_path, save_path,model)