from flask import Flask,render_template,request,redirect
from model import lm
import numpy as np

app=Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predictfun():
    if request.method=="POST":
        features=[int(i) for i in request.form.values()]
        final_features=[np.array(features)]
        predictions=lm.predict(final_features)
        finalPrediction=round(predictions[0],1)
        return render_template('predict.html',predictions=finalPrediction)
    else:
        return redirect('/')
app.run(debug=True)