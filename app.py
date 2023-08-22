import traceback
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from flask_cors import CORS
import re
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the CSV link from the environment variable
csv_link = os.getenv("CSV_LINK")

# Load the dataset
df = pd.read_csv(csv_link, low_memory=False)

# Select necessary columns
df = df[['RecipeId', 'Barcode', 'Name', 'CookTime', 'PrepTime', 'TotalTime', 'Calories',
       'FatContent', 'SaturatedFatContent', 'CholesterolContent',
       'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent',
       'ProteinContent', 'RecipeServings', 'DatePublished', 'RecipeInstructions', 'Images']]

# Handle fields with null values
df[df.isnull().any(axis=1)]

# Drop fields with NaN values
df = df.dropna()

# # Convert cook, prep, and total time to numeric values in minutes
df['CookTime'] = df['CookTime'].str.extract('(\d+)').astype(float) * 60
df['PrepTime'] = df['PrepTime'].str.replace('PT', '')
# df['PrepTime'] = df['PrepTime'].str.extract('(\d+)').astype(float)
df['TotalTime'] = df['TotalTime'].str.extract('(\d+)').astype(float) * 60

# Convert food contents to numeric
df["Calories"] = df["Calories"].astype(float)
df["FatContent"] = df["FatContent"].astype(float)
df["SaturatedFatContent"] = df["SaturatedFatContent"].astype(float)
df["CholesterolContent"] = df["CholesterolContent"].astype(float)
df["SodiumContent"] = df["SodiumContent"].astype(float)
df["CarbohydrateContent"] = df["CarbohydrateContent"].astype(float)
df["FiberContent"] = df["FiberContent"].astype(float)
df["SugarContent"] = df["SugarContent"].astype(float)
df["ProteinContent"] = df["ProteinContent"].astype(float)

# Convert Barcode to digits
df['Barcode'] = df['Barcode'].str.extract('(\d+)').astype(int)

# Fetch food data with calories less than 400
calorie_limit = 400
low_calories_data = df[df['Calories'] < calorie_limit]

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
calories_encoded = label_encoder.fit_transform(low_calories_data['Calories'])

# Feature extraction
features = low_calories_data[['FatContent', 'SaturatedFatContent', 'CholesterolContent',
                 'SodiumContent', 'CarbohydrateContent', 'FiberContent',
                 'SugarContent', 'ProteinContent']]

# Target variable
target = calories_encoded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Nearest Neighbors model
nn_model = KNeighborsRegressor(n_neighbors=5)
nn_model.fit(X_train, y_train)

# Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(features, target)

# Flask App Setup
app = Flask(__name__)
CORS(app)

def clean_data(data_string):
    if data_string == "character(0)":
        data_string = ""
        return data_string.split('", "')
    
    if data_string.startswith('c("') or data_string.endswith('")'):
        return data_string[3:-3].split('", "')
    
    return data_string.replace('\"', '').split('", "')   

@app.route("/food-recommendation", methods=['GET'])
# Food Recommendation Resource
def food_recommendation():
     try:
        barcode = request.args.get('barcode', type=int)
        bmi = request.args.get('bmi', type=int)
        
        if barcode is None:
                raise ValueError('Barcode is missing in the request.')
           
        if bmi is None:
                raise ValueError('BMI is missing in the request.')
                   
        # Find the nearest neighbors based on food contents
        barcode_data = df[df['Barcode'] == barcode][['FatContent', 'SaturatedFatContent',
                                                        'CholesterolContent', 'SodiumContent',
                                                        'CarbohydrateContent', 'FiberContent',
                                                        'SugarContent', 'ProteinContent']]
        
        food_data = df[df['Barcode'] == barcode].iloc[0]  # Extract the row as a Series
        
        if ((bmi < 25 ) or (food_data['Calories'] < calorie_limit)):
             response = {
                'food_data': {
                    'Name': food_data['Name'],
                    'PrepTime': food_data['PrepTime'],
                    'foodContents':{
                        'FatContent': food_data['FatContent'],
                        'SaturatedFatContent': food_data['SaturatedFatContent'],
                        'CholesterolContent': food_data['CholesterolContent'],
                        'SodiumContent': food_data['SodiumContent'],
                        'CarbohydrateContent': food_data['CarbohydrateContent'],
                        'FiberContent': food_data['FiberContent'],
                        'SugarContent': food_data['SugarContent'],
                        'ProteinContent': food_data['ProteinContent'],
                        'Calories': food_data['Calories']
                    },
                    'RecipeInstructions': clean_data(food_data['RecipeInstructions']),
                    'Images': clean_data(food_data['Images'])
               }   
            }
             return jsonify(response), 200
      
        nn_recipe_idxs = nn_model.kneighbors(barcode_data, n_neighbors=5, return_distance=False)[0]
        
        nn_recipes = [df.iloc[nn_recipe_idx] for nn_recipe_idx in nn_recipe_idxs]
        # Filter out recommendations with calories higher than 400
        nn_recipes = [recipe for recipe in nn_recipes if recipe['Calories'] < calorie_limit][0]

     

        # Use Naive Bayes to find a similar recipe with fewer calories
        similar_food_idx = nb_model.predict(barcode_data)[0]
        # print(similar_food_idx)
        similar_food = df[df['Calories'] < calorie_limit].iloc[similar_food_idx]
        response = {
               'food_data': {
                    'Name': food_data['Name'],
                    'PrepTime': food_data['PrepTime'],
                    'foodContents':{
                        'FatContent': food_data['FatContent'],
                        'SaturatedFatContent': food_data['SaturatedFatContent'],
                        'CholesterolContent': food_data['CholesterolContent'],
                        'SodiumContent': food_data['SodiumContent'],
                        'CarbohydrateContent': food_data['CarbohydrateContent'],
                        'FiberContent': food_data['FiberContent'],
                        'SugarContent': food_data['SugarContent'],
                        'ProteinContent': food_data['ProteinContent'],
                        'Calories': food_data['Calories'],
                    },
                    'RecipeInstructions': clean_data(food_data['RecipeInstructions']),
                    'Images': clean_data(food_data['Images'])
                },
                'nearest_recipe': {
                    'Name': nn_recipes['Name'],
                    'PrepTime': nn_recipes['PrepTime'],
                    'foodContents':{
                        'FatContent': nn_recipes['FatContent'],
                        'SaturatedFatContent': nn_recipes['SaturatedFatContent'],
                        'CholesterolContent': nn_recipes['CholesterolContent'],
                        'SodiumContent': nn_recipes['SodiumContent'],
                        'CarbohydrateContent': nn_recipes['CarbohydrateContent'],
                        'FiberContent': nn_recipes['FiberContent'],
                        'SugarContent': nn_recipes['SugarContent'],
                        'ProteinContent': nn_recipes['ProteinContent'],
                        'Calories': nn_recipes['Calories']
                    },
                    'RecipeInstructions': clean_data(nn_recipes['RecipeInstructions']),
                    'Images': clean_data(nn_recipes['Images'])
                },
                'similar_food_with_less_calories': {
                    'Name': similar_food['Name'],
                    'PrepTime': similar_food['PrepTime'],
                    'foodContents':{
                        'FatContent': similar_food['FatContent'],
                        'SaturatedFatContent': similar_food['SaturatedFatContent'],
                        'CholesterolContent': similar_food['CholesterolContent'],
                        'SodiumContent': similar_food['SodiumContent'],
                        'CarbohydrateContent': similar_food['CarbohydrateContent'],
                        'FiberContent': similar_food['FiberContent'],
                        'SugarContent': similar_food['SugarContent'],
                        'ProteinContent': similar_food['ProteinContent'],
                        'Calories': similar_food['Calories']
                    },
                    'RecipeInstructions': clean_data (similar_food['RecipeInstructions']),
                    'Images': clean_data (similar_food['Images'])
                }
            }
        return jsonify(response), 200
   
     except ValueError as e:
          error_traceback = traceback.format_exc()
          print({'error': str(e), 'traceback': error_traceback})
          return jsonify({'error': 'Food Data Not Found.'}), 400  # Bad Request
       
     except Exception as e:
          error_traceback = traceback.format_exc()
          print({'error': str(e), 'traceback': error_traceback})
          return jsonify({'error': 'An unexpected error occurred.'}), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='192.168.1.57', port=5000)