import pickle
from flask import Flask

app = Flask(__name__)

model_pickle = open("./artifacts/classifier.pkl", "rb")
clf = pickle.load(model_pickle)


@app.route("/ping", methods=['GET'])
def ping():
    return {"msg": "Hello World!"}
