# PoL_Anomaly_Detection

- Add details of how to setup opencv on a linux machine or just put a link on how to install that: https://www.pyimagesearch.com/2018/05/28/ubuntu-18-04-how-to-install-opencv/

- To work on the virtual environment execute the commande below: 
workon cv 


- Run video using this command: 
python3 main_video_detection.py --input videos/01.avi --output output/video01.avi --yolo yolo-coco

# Setup
- Get the YOLO v3 wieghts folder from daknet's website: https://pjreddie.com/darknet/yolo/
OR 
Just execute this command in your terminal:
wget https://pjreddie.com/media/files/yolov3.weights

# Datasets:
Download datasets from here
- 
ShanghaiTech Campus dataset (Anomaly Detection): https://svip-lab.github.io/dataset/campus_dataset.html

- UCSD Anomaly Detection Dataset: http://www.svcl.ucsd.edu/projects/anomaly/dataset.html 

- Avenue Dataset for Abnormal Event Detection: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html

Then put them in folder ...

# Repository hierarchy 