from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predictfun():
    if request.method=="POST":
        features=[int(i) for i in request.form.values()]
        final_features=[np.array(features)]
        predictions=model.predict(final_features)
        finalPrediction=round(predictions[0],1)
        return render_template('predict.html',predictions=finalPrediction)
    else:
        return redirect('/')
app.run(debug=True)