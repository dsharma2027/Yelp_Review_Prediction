# Importing the required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import pickle
from flask import Flask, render_template, request
from flasgger import Swagger
import numpy as np


# Importing the Pickle File
pkl_import=open('MNB_classifier.pkl','rb')
classifier=pickle.load(pkl_import)

#Initialising the  Flask and Swagger
app=Flask(__name__,template_folder="templates")
Swagger(app)

# Creating the Base Route
@app.route('/',methods=['GET'])
def welcome_message():
    try:
        return render_template("home.html"),200
    except Exception as e:
        return "something went wrong",400

#Creeating a route for Prediction
@app.route('/predict',methods=['GET'])
def predict():
   
    """Testing of Review prediction .
    ---
    parameters:  
      - name: review
        in: query
        type: string
        required: true
    responses:
        200:
            description: The response is 
        
    """

    review = request.args.get("review")

    result = classifier.predict([review])
    return f"The result are as follows {result}"
    # if result in [0,"0"] :  return "Fake Bank Note",200
    # return "Valid bank note",200


# Creating a route for Prediction using test file
@app.route('/predict_through_file',methods=["POST"])
def predict_file():
    """Testing of prediction using Test file
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The response of file is         
    """

    df = pd.read_csv(request.files.get("file"))
    print("-"*30+" File details "+"-"*30)
    print(df.shape)
    print(df.head())
    print("-"*70)
    
    result = classifier.predict(df.values.ravel())
    print(result.shape)
    print(type(result))
    print(len(list(result)))
    return f"The result are as follows {list(result)} and the length is {len(list(result))}"


# Main
if __name__ == "__main__":
    app.run(debug=True)