# Lendwise AI - Credit Score Predictor #
## Project Overview ##

- Lendwise AI is a simple web application that predicts a user's credit score based on a set of financial and behavioral inputs. It uses a machine learning model to generate a score and a risk category, providing a quick and easy way for businesses to understand their creditworthiness.

- This project serves as a demonstration of a full-stack machine learning application, with a Python backend handling the model predictions and a simple HTML/CSS frontend for user interaction.

## Features ##

- Credit Score Prediction: Predicts a credit score using a trained machine learning model.

- Risk Categorization: Classifies the predicted score into a risk category (Low, Medium, or High Risk).

- Informational Pages: Includes an "About" page to explain credit scores and their importance.

## Project Structure ##

- app.py: The main Python backend server that serves the web pages and handles API requests for credit score prediction.

- index.html: The main page with the input form for users to enter their data.

- check.html: The results page that displays the predicted credit score and risk category.

- about.html: An informational page about credit scores.

- dataset_2.csv: The dataset used to train the machine learning models.

- model.pkl: The saved machine learning pipeline. This file contains the trained XGBoost Regressor and the preprocessing steps, which is the core of the prediction logic.

- kmeans.pkl: The saved K-Means clustering model. This is used to group the predicted numerical credit scores into a specific number of clusters, which are then used to define the risk categories.

- scaler.pkl: The saved StandardScaler model. It's essential for preprocessing new user input by scaling the numerical features in the exact same way as the data used for model training.

- risk_mapping.pkl: A file containing a dictionary that maps the numerical cluster IDs from the kmeans.pkl model to the descriptive risk categories (e.g., "Low Risk").

- features.pkl: This file stores the list of feature names used during training, which is crucial for ensuring that new data is presented to the model in the correct order.

## Installation & Setup ##

To run this project locally, follow these steps:

**Clone the repository:**

>git clone https://github.com/your-username/your-repo-name.git
>cd your-repo-name

**Set up a virtual environment (recommended):**

>python -m venv venv
