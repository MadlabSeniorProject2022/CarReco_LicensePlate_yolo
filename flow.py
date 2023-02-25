import torch
import os
import cv2
from roboflow import Roboflow
import json
import time

# Load License plate Model
license_plate_model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'LPD150.pt', force_reload=True, trust_repo=True)

# Load Character Segmentation Model From Roboflow
rf = Roboflow(api_key="aCtlUV2bPblR04ocOyk5")
project = rf.workspace().project("lru-license-plate")
character_segmentation_model = project.version(1).model
 
# Load Json for character mapping
with open('char_map.json') as json_file:
    char_map = json.load(json_file)

def find_plate (img_path):
  img = cv2.imread(img_path)
  result = license_plate_model(img_path)
  coor = result.xyxy[0]
  if len(coor) == 0:
     return None
  
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
       result_test = result_test + char_map[c["class"]]
    return result_test

def run_flow (img_path):
    plate = find_plate(img_path)
    if plate == None:
        return "ไม่พบป้ายทะเบียน"
    isExist = os.path.exists("./plates")
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs("./plates")
    current_time = int(time.time()*10000)
    cv2.imread(f'./plates/{current_time}.jpg', plate)
    license_num = read_char(f'./plates/{current_time}.jpg')
    return license_num