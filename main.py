from flask import Flask, jsonify
from predict import predict

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict(data):
    try:
        prediction = predict(data)
        return jsonify(prediction)
    except:
        return jsonify({'error nnig'}, 400)


if __name__ == '__main__':
    app.run(debug=True)
