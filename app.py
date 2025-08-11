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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.get_json(force=True) or {}
        forecast = get_forecast(user_input)
        return jsonify(forecast), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
