# CluckCare
# Web Application Overview

🐔🐥🐓 CluckCare is a website harnessing the power of deep learning convolutional neural networks (CNN-VGG) to predict chicken diseases from uploaded images of their excretions. Our simple yet effective approach aims to assist poultry farmers and veterinarians in promptly identifying potential health issues.

# Try out the App

https://cluckcare.onrender.com

The app is hosted on render cloud with the help of a docker image for you to try it out.

If the website is not responding immediately please give it a few minutes since the container instances will scale down to zero after a period of in activity.

# Run the Web App in a docker container
   - Just pull the following Docker image from Docker Hub

      ```docker.io/senthilshunmugam/cluckcare:1.0```


   - If you are feeling adventurous and want to modify the code your self then you can clone the repository and use the DockerFile to build your own version of the image using the following cmd.

     ```docker build -t <imagename> .```
     
# Run the Web App Locally 

   - Clone this repo and install the requirements.txt
   - Run the `app.py` file

# CNN-VGG Model 

Convolutional Neural Network - Visual Geometry Group (CNN-VGG) model implemented in TensorFlow/Keras for image classification. The model is trained on a dataset consisting of images belonging to three different classes.

![image](https://github.com/K-Senthil-Shunmugam/CluckCare/assets/113205555/a86fe681-4b1e-43a5-a262-186a015bbd4c)

***Model Metrics***


**Accuracy: 96.50%**


**Precision: 97.00%**


**Recall: 97.00%**


**F1 Score: 97.00%**

![image](https://github.com/K-Senthil-Shunmugam/CluckCare/assets/113205555/e0509a92-aeed-496f-89fc-8ccad3ae86d2)

![image](https://github.com/K-Senthil-Shunmugam/CluckCare/assets/113205555/236caf69-983d-4e12-a16f-9f3b4091f64f)

![image](https://github.com/K-Senthil-Shunmugam/CluckCare/assets/113205555/ff00b6e2-c942-443c-b6d8-53577f68f8df)


## Training the Sentiment Analysis Model (Model Training.ipynb)

To train the CNN-VGG model for your own dataset, follow these instructions:

1. **Ensure Dependencies:**
   - Make sure you have Jupyter Notebook installed along with the required Python libraries mentioned in the provided `Model Training.ipynb` file.

2. **Prepare Your Dataset:**
   - you can download the Dataset used in this project either from Kaggle
https://www.kaggle.com/datasets/ramkishore1/bird-disease-dataset
   - Load the dataset from Kaggle or Replace it with your own dataset.
   - I have dropped the NCD class in my version of the code since it has a very low number of samples.

4. **Run the Notebook:**
   - Open the `Model Training.ipynb` notebook in Jupyter Notebook.
   - Execute each cell in the notebook sequentially to load the dataset, preprocess the data, train the model, and save the trained model.

5. **Adjust Hyperparameters (Optional):**
   - You can adjust the hyperparameters such as callbacks, learning rate, batch size, and number of epochs according to your requirements.

6. **Save the Trained Model:**
   - After training, the model will be saved as `model.h5` in the same directory.
   - You can use this trained model for inference in the web app `app.py` provided in this repository.

7. **Evaluate Model Performance (Optional):**
   - Optionally, you can evaluate the performance of your trained model on a separate test dataset to assess its accuracy and other metrics.

8. **Customize as Needed:**
   - Feel free to customize the notebook or extend the functionality based on your specific requirements.
   - You can also integrate additional features or improve the model architecture for better performance.
