from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict

app = Flask(__name__)
cors = CORS(app)


@app.route("/predict", methods=["POST"])
def predict_route():
    result = predict(request.get_json())[0]

    return jsonify(
        {
            "homeWin": str(result[0] * 100),
            "draw": str(result[1] * 100),
            "awayWin": str(result[2] * 100),
        }
    )


if __name__ == '__main__':
    app.run(debug=True)
