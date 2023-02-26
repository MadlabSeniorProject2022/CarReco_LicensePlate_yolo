import torch
import os
import cv2
from roboflow import Roboflow
import json
import time

from google.cloud import storage


# Load License plate Model
license_plate_model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'LPD150.pt', force_reload=True, trust_repo=True)

# Load Character Segmentation Model From Roboflow
rf = Roboflow(api_key="aCtlUV2bPblR04ocOyk5")
project = rf.workspace().project("lru-license-plate")
character_segmentation_model = project.version(1).model
 
# Load Json for character mapping
with open('char_map.json') as json_file:
    char_map = json.load(json_file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r'./carrgclassification-857e3c375cdd.json'

def upload_cs_file(bucket_name, source_file_name, destination_file_name): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    blob = bucket.blob(destination_file_name)
    blob.upload_from_filename(source_file_name)

import datetime

def get_cs_file_url(bucket_name, file_name, expire_in=datetime.datetime.now() + datetime.timedelta(4000)): 
    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    url = bucket.blob(file_name).generate_signed_url(expire_in)

    return url

def cloud_image (bucket_name, source_file_name, destination_file_name):
    upload_cs_file(bucket_name, source_file_name, destination_file_name)
    url = get_cs_file_url(bucket_name, destination_file_name)
    return url

def find_plate (img_path):
  img = cv2.imread(img_path)
  result = license_plate_model(img_path)
  coor = result.xyxy[0]
  if len(coor) == 0:
     return 404
  
  xmin = min([i[1] for i in coor])
  xmax = max([i[3] for i in coor])
  ymin = min([i[0] for i in coor])
  ymax = max([i[2] for i in coor])
  crop_img = img[ int(xmin) : int(xmax), int(ymin): int(ymax)] #crop with range that cover point that possible it's license plate
  return crop_img

def read_char (img_path):
    res = character_segmentation_model.predict(img_path, confidence=80, overlap=30).json()["predictions"] #get prediction
    res.sort(key=lambda x: x["x"])
    accepted_res = []
    candidate = []
    for c in res:
        #candidate.append(c)
        if len(candidate) > 0:
            if c["x"] > candidate[-1]["x"] + (candidate[-1]["width"]/2):
                conf = [i["confidence"] for i in candidate]
                index = conf.index(max(conf))
                accepted_res.append(candidate[index])
                candidate.clear()
        candidate.append(c)
    conf = [i["confidence"] for i in candidate]
    index = conf.index(max(conf))
    accepted_res.append(candidate[index])
    result_test = ""
    for c in accepted_res:
       result_test = result_test + char_map[c["class"]]
    return result_test

def run_flow (img_path):
    plate = find_plate(img_path)
    if type(plate) == int:
        return ("ไม่พบป้ายทะเบียน", None)
    isExist = os.path.exists("./plates")
    if not isExist:
      # Create a new directory because it does not exist
      os.makedirs("./plates")
    current_time = int(time.time()*10000)
    cv2.imwrite(f'./plates/{current_time}.jpg', plate)
    plate_img_path = cloud_image('images-bucks', f'./plates/{current_time}.jpg', f'{current_time}-plate.jpg')
    license_num = read_char(f'./plates/{current_time}.jpg')
    if license_num == "":
       license_num = "ไม่สามารถอ่านเลขทะเบียนได้"
    return (license_num, plate_img_path)