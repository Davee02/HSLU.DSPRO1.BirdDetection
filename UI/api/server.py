from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # we will get the file from the request
    file = request.files['file']
    # convert that to bytes
    img_bytes = file.read()

if __name__ == '__main__':
    app.run()