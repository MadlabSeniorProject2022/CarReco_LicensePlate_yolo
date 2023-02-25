import torch
import os
import cv2
from roboflow import Roboflow

#Load License plate Model
license_plate_model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'LPD150.pt', force_reload=True, trust_repo=True)

#Load Character Segmentation Model From Roboflow
rf = Roboflow(api_key="aCtlUV2bPblR04ocOyk5")
project = rf.workspace().project("lru-license-plate")
character_segmentation_model = project.version(1).model

def find_plate (img_path):
  img = cv2.imread(img_path)
  result = license_plate_model(img_path)
  coor = result.xyxy[0]
  xmin = min([i[1] for i in coor])
  xmax = max([i[3] for i in coor])
  ymin = min([i[0] for i in coor])
  ymax = max([i[2] for i in coor])
  crop_img = img[ int(xmin) : int(xmax), int(ymin): int(ymax)] #crop with range that cover point that possible it's license plate
  return crop_img

def read_char (img_path):
    res = license_plate_model.predict(img_path, confidence=60, overlap=30).json()["predictions"] #get prediction
    res.sort(key=lambda x: x["x"])
    result_test = ""
    for c in res:
       pass
    return result_test