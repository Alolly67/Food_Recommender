import traceback
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from flask_cors import CORS

# Load the dataset
df = pd.read_csv('recipes.csv', low_memory=False)

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
df['PrepTime'] = df['PrepTime'].str.extract('(\d+)').astype(float) * 60
df['TotalTime'] = df['TotalTime'].str.extract('(\d+)').astype(float) * 60

# Convert Barcode to digits
df['Barcode'] = df['Barcode'].str.extract('(\d+)').astype(int)

# Encode the target variable using LabelEncoder
label_encoder = LabelEncoder()
calories_encoded = label_encoder.fit_transform(df['Calories'])

# Feature extraction
features = df[['FatContent', 'SaturatedFatContent', 'CholesterolContent',
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
    data_to_list = [data_str.strip('" ') for data_str in data_string[3:-3].split('", "')]
    return data_to_list
    

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
        
        if (bmi < 25):
             response = {
                'food_data': {
                    'foodContents':{
                        'Name': food_data['Name'],
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
      
        nn_recipe_idx = nn_model.kneighbors(barcode_data, n_neighbors=1, return_distance=False)[0][0]

     

        # Use Naive Bayes to find a similar recipe with fewer calories
        calorie_limit = 400
        similar_food_idx = nb_model.predict(barcode_data)[0]
        
        response = {
               'food_data': {
                    'foodContents':{
                        'Name': food_data['Name'],
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
                    'foodContents':{
                        'Name': df.iloc[nn_recipe_idx]['Name'],
                        'FatContent': df.iloc[nn_recipe_idx]['FatContent'],
                        'SaturatedFatContent': df.iloc[nn_recipe_idx]['SaturatedFatContent'],
                        'CholesterolContent': df.iloc[nn_recipe_idx]['CholesterolContent'],
                        'SodiumContent': df.iloc[nn_recipe_idx]['SodiumContent'],
                        'CarbohydrateContent': df.iloc[nn_recipe_idx]['CarbohydrateContent'],
                        'FiberContent': df.iloc[nn_recipe_idx]['FiberContent'],
                        'SugarContent': df.iloc[nn_recipe_idx]['SugarContent'],
                        'ProteinContent': df.iloc[nn_recipe_idx]['ProteinContent'],
                        'Calories': df.iloc[nn_recipe_idx]['Calories']
                    },
                    'RecipeInstructions': clean_data(df.iloc[nn_recipe_idx]['RecipeInstructions']),
                    'Images': clean_data(food_data['Images'])
                },
                'similar_food_with_less_calories': {
                    'foodContents':{
                        'Name': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['Name'],
                        'FatContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['FatContent'],
                        'SaturatedFatContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['SaturatedFatContent'],
                        'CholesterolContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['CholesterolContent'],
                        'SodiumContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['SodiumContent'],
                        'CarbohydrateContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['CarbohydrateContent'],
                        'FiberContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['FiberContent'],
                        'SugarContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['SugarContent'],
                        'ProteinContent': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['ProteinContent'],
                        'Calories': df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['Calories']
                    },
                    'RecipeInstructions': clean_data(df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['RecipeInstructions']),
                    'Images': clean_data(df[df['Calories'] < calorie_limit].iloc[similar_food_idx]['Images'])
                }
            }
        return jsonify(response), 200
   
     except ValueError as e:
          error_traceback = traceback.format_exc()
          return jsonify({'error': str(e), 'traceback': error_traceback}), 400  # Bad Request
       
     except Exception as e:
          error_traceback = traceback.format_exc()
          return jsonify({'error': str(e), 'message': 'An unexpected error occurred.', 'traceback': error_traceback}), 500  # Internal Server Error


if __name__ == '__main__':
    app.run(debug=True, host='192.168.14.226', port=5000)