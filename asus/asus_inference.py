import base64
from flask import Flask, request, jsonify
import logging
import json
from model_handler import ModelHandler 
from PIL import Image
import io
app = Flask(__name__)
logger = logging.getLogger('flask.app')
model_handler = ModelHandler()
model_handler.init()

@app.route("/yolov7", methods=['GET'])

def yolov7():
    return "Yolov7 is serving!"


@app.route("/yolov7/detect", methods=['POST'])
def detect():
    print("yolov7()", flush=True)
    body = request.get_json()
    if 'image' not in body:
        return jsonify({"status": "error",'results': 'no file'}), 400
    thresh = 0.5
    if "thresh" in body:
        thresh = body["thresh"]
        print("thresh: %s" % thresh)
    buf = io.BytesIO( base64.b64decode(body['image'].encode('utf8')))
    image = Image.open(buf)
    results = model_handler.detect(image,thresh)
    res = {"status": "success", "results": results}
    return jsonify(res), 200








if __name__ == "__main__":
    # Only for debugging while developing

    app.run(host="0.0.0.0", debug=True, port=9999)
