from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import joblib
import os

# Global variables for models
model = None
feature_names = None
kmeans = None
scaler = None
risk_mapping = None

class RequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
    
    def do_GET(self):
        try:
            if self.path == '/':
                with open('index.html', 'rb') as f:
                    self._set_headers()
                    self.wfile.write(f.read())
            elif self.path == '/check.html':
                with open('check.html', 'rb') as f:
                    self._set_headers()
                    self.wfile.write(f.read())
            elif self.path == '/features':
                self._set_headers('application/json')
                response = {"features": feature_names}
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_error(404, "File not found")
        except Exception as e:
            self.send_error(500, str(e))
    
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                user_df = pd.DataFrame([data], columns=feature_names)
                
                # Predict credit score
                predicted_score = model.predict(user_df)[0]
                
                # Predict risk category
                scaled_score = scaler.transform([[predicted_score]])
                risk_num = kmeans.predict(scaled_score)[0]
                risk_category = risk_mapping[risk_num]
                
                response = {
                    "credit_score": float(predicted_score),
                    "risk_category": risk_category,
                    "risk_num": int(risk_num)
                }
                
                self._set_headers('application/json')
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404, "Not Found")

def load_or_train_models():
    global model, feature_names, kmeans, scaler, risk_mapping
    
    # Try to load saved models
    if all(os.path.exists(f) for f in ['model.pkl', 'kmeans.pkl', 'scaler.pkl']):
        model = joblib.load('model.pkl')
        kmeans = joblib.load('kmeans.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('features.pkl')
        risk_mapping = joblib.load('risk_mapping.pkl')
        return
    
    # Train new models
    df = pd.read_csv("dataset_2.csv")
    df.columns = df.columns.str.strip()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Train credit score model
    df = df.drop(columns=["Business_ID", "Risk_Category"], errors='ignore')
    X = df.drop(columns=["Credit_Score"], errors='ignore')
    y = df["Credit_Score"]
    
    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(exclude=np.number).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42))
    ])
    model.fit(X, y)
    feature_names = X.columns.tolist()
    
    # Train risk classifier
    credit_scores = df[["Credit_Score"]].values
    scaler = StandardScaler()
    credit_scores_scaled = scaler.fit_transform(credit_scores)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    risk_nums = kmeans.fit_predict(credit_scores_scaled)
    
    # Create risk mapping
    risk_stats = df.groupby(risk_nums)["Credit_Score"].mean()
    risk_mapping = {
        risk_stats.idxmax(): "Low Risk",
        risk_stats.idxmin(): "High Risk"
    }
    risk_mapping = {k: risk_mapping.get(k, "Medium Risk") for k in range(3)}
    
    # Save models
    joblib.dump(model, 'model.pkl')
    joblib.dump(kmeans, 'kmeans.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'features.pkl')
    joblib.dump(risk_mapping, 'risk_mapping.pkl')

if __name__ == '__main__':
    load_or_train_models()
    server = HTTPServer(('localhost', 8000), RequestHandler)
    print("Server running at http://localhost:8000")
    server.serve_forever()