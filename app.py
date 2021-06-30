# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import sklearn
from PIL import Image
import pickle
import tensorflow as tf



# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page


@ app.route('/')
def home():
    title = 'Human Activity Recognition'
    return render_template('index.html', title=title)

# render crop recommendation form page
@ app.route('/', methods=['POST'])
def home_submit():
    ACTIVITIES = {0: 'WALKING',1: 'WALKING_UPSTAIRS',2: 'WALKING_DOWNSTAIRS',3: 'SITTING',4: 'STANDING',5: 'LAYING'}
    body_acc_x = float(request.form['1'])
    body_acc_y = float(request.form['2'])
    body_acc_z = float(request.form['3'])
    body_gyro_x=float(request.form['4'])
    body_gyro_y=float(request.form['5'])
    body_gyro_z=float(request.form['6'])
    total_acc_x=float(request.form['7'])
    total_acc_y=float(request.form['8'])
    total_acc_z=float(request.form['9'])
    input_array=[body_acc_x,body_acc_y,body_acc_z,body_gyro_x,body_gyro_y,body_gyro_z,total_acc_x,total_acc_y,total_acc_z]
    lst=[]
    for i in range(128):
        lst.extend(input_array)
    inp=np.array(lst)
    l=inp.reshape(1,128,9)
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model =  tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    result=loaded_model.predict(l)
    prediction = (np.argmax(result[0], axis=0))
    return render_template('result.html',prediction = ACTIVITIES[prediction])

# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=True)
