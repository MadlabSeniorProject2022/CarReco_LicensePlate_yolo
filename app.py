from flask import *
from flow import run_flow

import os 
import time
import io

from PIL import Image

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict ():
    if request.method != "POST":
        return jsonify({'msg': 'notfound', 'predicted': None})

    if request.files.get("image"):
        im_file = request.files["image"]
        im_bytes = im_file.read()
        path = f"./to_predict/{int(time.time() * 1000000)}.jpg"
        im_file = request.files["image"]
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        plate_num, url = run_flow(im)

    return jsonify({'msg': 'success', 'plate_num': plate_num, 'plate_url': url})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))