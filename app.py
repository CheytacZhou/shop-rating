import numpy as np
from flask import Flask, render_template, request
import pickle
import rating
from rating import rating

# Initialize the flask APP
app = Flask(__name__)

# Load the model
# rating.main() # objects are being pickled with main_module as the top-level
# with open("rating.pkl", "rb") as f:
    # mod = pickle.load(f)
mod = rating()

# Default page for our web-app
@app.route("/")
def home():
    return render_template("index.html")

# Use the predict button in our web-app
@app.route("/predict", methods = ["POST"])
def predict():
    # For rendering results on HTML GUI
    features = [list(map(eval, x.split())) for x in request.form.values()] # features_list = [float(x) for x in request.form.values()]
    # features = mod.preprocess(features_list)
    score_dim, score_overall = mod.predict(features)
    return render_template(
        "index.html", 
        pred_overall = "零售商户综合得分为: {}".format(score_overall),
        pred_dim = "零售商户维度得分为: 地理位置({}) 店铺属性({}) 经管指标({}) 社媒流量({}) 竞争潜力({}) 用户忠诚({})".format(
            score_dim[0], score_dim[1], score_dim[2], score_dim[3], score_dim[4], score_dim[5]
        )
    )

# Use the model api for prediction
@app.route("/predict_api", methods = ["POST"])
def predict_api():
    if mod:
        json_ = request.json
        query = list(json_[0].values())
        score_dim, score_overall = mod.predict(query)
        return jsonify({"维度得分": score_dim, "综合得分": score_overall})
    else:
        print("Train the model first")
        return("No model here to use")

if __name__ == "__main__":
    app.run(debug = True)
