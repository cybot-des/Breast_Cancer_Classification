from flask import Flask,render_template,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('static/model/cancer_classification_model.pkl','rb'))

def predict(feature_arr):
    feature_val_arr = np.array(feature_arr).reshape(1,9)
    result = model.predict(feature_val_arr)
    return result[0]


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result',methods=['POST','GET'])
def result():
    if request.method == 'POST':
        feature_arr = request.form.to_dict()
        feature_arr = list(feature_arr.values())
        feature_arr = list(map(int, feature_arr))
        result = predict(feature_arr)
        if result == 2:
            prediction = "Cancer is Benign"
        else:
            prediction = "Cancer is Malignant"
        return render_template('result.html', result = prediction)


if __name__ == '__main__':
    app.run(debug=True)