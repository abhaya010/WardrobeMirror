import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib

def load_image_features(image_features_path):
    df = pd.read_csv(image_features_path)
    return df

def prepare_data(df):
    X = df.drop(columns=['filename'])  # Features
    return X

def train_knn(X, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
    knn.fit(X)
    return knn

def save_model(model, model_path='/home/abhaya/Desktop/fafaFInal/myWork/src/models/knn_model1.pkl'):
    joblib.dump(model, model_path)
    print(f"\nModel saved successfully at {model_path}!")

def main():
    image_features_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'
    
    df = load_image_features(image_features_path)
    
    X = prepare_data(df)
    
    knn = train_knn(X, n_neighbors=5)
    
    save_model(knn)

if __name__ == "__main__":
    main()