from flask import Flask, render_template, request
import pickle
import pandas as pd
import scipy
import numpy as np

app = Flask(__name__)
file = open('model.pkl', 'rb')
regr = pickle.load(file)
file.close()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        myDict = request.form
        engine = float(myDict['Engine'])
        input_size = [engine]
        test_y_ = regr.predict([input_size])[0][0]
        return render_template('result.html', EMI=round(test_y_))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
