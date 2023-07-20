from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

@app.route("/hello")
def hello():
    return "Hello, World!"

@app.route("/echo")
def echo():
    params = request.args
    return jsonify(params)

@app.route("/upload_image", methods=["POST"])
def upload_image():
    image = request.files.get("image")
    
    if image is None:
        return jsonify({"error": "No image received."}), 400
    
    try:
        img = Image.open(BytesIO(image.read()))
        width, height = img.size
        return jsonify({"width": width, "height": height})
    except Exception as e:
        return jsonify({"error": "Failed to process the image.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
