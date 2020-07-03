from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import pickle


app = Flask(__name__)
model = open('stock_rf.pkl','rb')
forest = pickle.load(model)

countvect = open('cv.pkl', 'rb')
cv = pickle.load(countvect)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        sentiments = request.form["sentiments"]
        data = [sentiments]
        vect = cv.transform(data).toarray()

        predictions = forest.predict(vect)
        output = predictions[0]

        if output == 1:
            return render_template("home.html", prediction = "Stock price will increase")
        else:
            return render_template("home.html", prediction = "Stock price will not increase")

if __name__ == "__main__":
    app.run(debug=True)

