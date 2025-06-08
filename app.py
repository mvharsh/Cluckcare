from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

app = Flask(__name__)

# Define the image size
img_size = (150, 150)

# Load the saved model
loaded_model = models.load_model('model.h5')

# Define class names
class_name = ['cocci' ,'salmo' ,'healthy']

# Function to predict class of an image
def predict_class(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_size[0], img_size[1]))
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  

    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_name[predicted_class_index]
    
    return predicted_class

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/healthy')
def healthy():
    return render_template('healthy.html')

@app.route('/salmo')
def salmo():
    return render_template('salmo.html')


@app.route('/cocci')
def cocci():
    return render_template('cocci.html')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No file part")
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file:
        filename = secure_filename(file.filename)
        file.save(filename)
        predicted_class = predict_class(filename, loaded_model)
        os.remove(filename)
        # Redirect to different pages based on prediction
        if predicted_class == "healthy":
            return redirect(url_for('healthy'))
        elif predicted_class == "salmo":
            return redirect(url_for('salmo'))
        elif predicted_class == "cocci":
            return redirect(url_for('cocci'))
        else:
            return render_template('index.html', prediction="Unknown class")
    else:
        return render_template('index.html', prediction="Prediction failed")



if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0")

