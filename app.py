from flask import *
from flow import run_flow

import os 
import time
import io
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict ():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        isExist = os.path.exists("./to_predict")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs("./to_predict")
        path = f"./to_predict/{int(time.time() * 1000000)}.jpg"
        file_bytes = np.asarray(bytearray(io.BytesIO(im_bytes).read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv2.imwrite(path, img)
        plate_num, url = run_flow(path)

    return jsonify({'msg': 'success', 'plate_num': plate_num, 'plate_url': url})
 
@app.route("/")
def ping ():
    return "Plate Recpgnition is working"
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))