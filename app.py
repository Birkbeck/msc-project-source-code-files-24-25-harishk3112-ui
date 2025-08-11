from flask import Flask, jsonify, render_template, request
from model import get_forecast

app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route('/.well-known/<path:subpath>')
def well_known(subpath):
    return ('', 204)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    forecast = get_forecast(user_input)
    return jsonify(forecast)

if __name__ == '__main__':
    app.run(debug=True)
