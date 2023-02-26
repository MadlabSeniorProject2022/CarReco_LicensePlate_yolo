# CarReco_LicensePlate_yolo

Flask api service for Thailand license plates in car number reading. Base on YoloV7 a Convolution Neural Network.

# Concept and Step
1. We train new one license plate detection model from YoloV7 pretrained model to find and crop Thai license plate in front view car picture.
2. To read text in plates, We use a Character segmentation model that train form YoloV7 pretrained model with Thai License plate data from Roboflow (https://universe.roboflow.com/lru/lru-license-plate/dataset/1?fbclid=IwAR2ziGY9P6I71sWdG4ShLnFzYGSV1PUjtyGhYEicHvy5M6qOFC1znpNXLrg) That give us lists of character class and coordinate.
3. To make it readable in simple form, We arrange character list by x-coordinate and map any class with real word.

# Contributor
1. Thanick Chongtrakul
2. Phatcharapol Mungkung

# Special Thanks
LRU License Plate Computer Vision Project (https://universe.roboflow.com/lru/lru-license-plate/dataset/1?fbclid=IwAR2ziGY9P6I71sWdG4ShLnFzYGSV1PUjtyGhYEicHvy5M6qOFC1znpNXLrg) for Character Segmentation Model
