# Colorization and Style Transfer
 Project Machine learning in AI training course Vinbigdata. Working in group to make an end-to-end system to colorize gray image, then apply style transfer.

link model: https://drive.google.com/file/d/1HbBlhSZOlJNQ_uveVP0xoD7Xx2v4CyOJ/view?usp=sharing

# install requirements
pip install -r requirements.txt

# run server interminal 1
python main_fastapi.py

# run font end in terminal 2
python font_end.py

streamlit run font_end.py


# Demo colorization
<p align="center">
  <img src="resource_images/flower.jpg" width=400 height=1000 >
  <img src="resource_images/wolf.jpg" width=400 height=1000 >
  <img src="resource_images/HCM.jpg" width=400 height=1000 ><br/>
  <i>colorization demo (from top to bottom: gray image, colorized image 1, colorized image 2)</i>
</p>

# Demo style transfer

<p align="center">
  <img src="resource_images/style.png" width=400 height=500 >
  <img src="resource_images/horse.png" width=400 height=500 >
  <img src="resource_images/horse_style.png" width=400 height=500 ><br/>
  <i>style transfer demo (from left to right: style, original image, style transfered image)</i>
</p>


# Demo video colorization then style transfer
<p align="center">
  <img src="resource_images/demo_video.gif" width=600><br/>
  <i>Result</i>
</p>


# Acknowledge
This project is done by 5-mem group.
