import pickle
from tokenize import Name
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, Response

app = Flask(__name__,template_folder='templates')

data = pd.read_csv("Placement_Data_Full_Class.csv")
model = pickle.load(open("campus_data(KNN).pkl",'rb'))
#data = pd.read_csv("Placement_Data_Full_Class.csv")
model_salary = pickle.load(open('campus_data(linear_reg).pkl','rb'))


@app.route('/')
def index():
    ssc_p = sorted(data['ssc_p'].unique())
    mba_p = sorted(data['mba_p'].unique())
    return render_template('index.html', ssc_p=ssc_p, mba_p=mba_p)

@app.route('/predict', methods=['POST'])
def predict():
    ssc_p = float(request.form.get('ssc_p'))
    mba_p = float(request.form.get('mba_p'))
    Name = request.form.get('Name')
    Email = request.form.get('Email')
    print(ssc_p,mba_p,Name,Email)
    
    final_features = pd.DataFrame([[ssc_p,mba_p]], columns=['ssc_p','mba_p'])        
    prediction = model.predict(final_features)
    if(prediction==1):
        prediction = "Status : Placed"
    else:
         prediction = "Status : Not Placed"
    return render_template("index.html", a=str(prediction),n=Name)

@app.route('/pred_salary', methods=['POST'])
def salary():
    ssc_p = sorted(data['ssc_p'].unique())
    mba_p = sorted(data['mba_p'].unique())
    hsc_p = sorted(data['hsc_p'].unique())
    hsc_s = sorted(data['hsc_s'].unique())
    degree_p = sorted(data['degree_p'].unique())
    etest_p = sorted(data['etest_p'].unique())
    return render_template('salary.html',ssc_p=ssc_p, mba_p=mba_p, hsc_p=hsc_p, hsc_s=hsc_s, degree_p=degree_p, etest_p=etest_p)

@app.route('/salary', methods=['POST'])
def check():
    ssc_p = float(request.form.get('ssc_p'))
    hsc_p = float(request.form.get('hsc_p'))
    hsc_s = request.form.get('hsc_s')
    degree_p = float(request.form.get('degree_p'))
    etest_p = float(request.form.get('etest_p'))
    mba_p = float(request.form.get('mba_p'))
    print(ssc_p,hsc_p,hsc_s,degree_p,etest_p,mba_p)
    
    check_salary = pd.DataFrame([[ssc_p,hsc_p,hsc_s,degree_p,etest_p,mba_p]], columns=['ssc_p','hsc_p','hsc_s','degree_p','etest_p','mba_p'])        
    prediction = model_salary.predict(check_salary)

    return render_template('salary.html', c = prediction)

if __name__ == "__main__":
    app.run(debug=True)
