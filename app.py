from flask import Flask, jsonify, render_template, request
from model import get_forecast

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")