import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import ResNet50
from sklearn.neighbors import NearestNeighbors

# Load the pre-trained ResNet50 model for feature extraction
def load_feature_extraction_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    return model

# Preprocess an image from a local file path
def preprocess_image_from_path(image_path):
    if not image_path:
        return None
    
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to match ResNet50 input size
        img_array = np.array(img)

        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]  # Remove alpha channel if present

        img_array = img_array.astype('float32')  
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error processing image from {image_path}: {e}")
        return None

# Extract features from a local image path
def extract_features_from_path(image_path, model):
    img_array = preprocess_image_from_path(image_path)
    if img_array is not None:
        if img_array.shape == (1, 224, 224, 3):
            features = model.predict(img_array)
            return features.flatten()
        else:
            print(f"Unexpected image shape: {img_array.shape}")
            return None
    else:
        print(f"Invalid image from path: {image_path}")
        return None

# Load the KNN model and feature vectors
def load_knn_model_and_features(knn_model_path, feature_vectors_path):
    knn_model = joblib.load(knn_model_path)
    feature_vectors_df = pd.read_csv(feature_vectors_path)
    return knn_model, feature_vectors_df

# Find similar images using the KNN model
def find_similar_images(query_features, knn_model, feature_vectors_df, n_neighbors=5):
    X = feature_vectors_df.drop(columns=['filename'])
    distances, indices = knn_model.kneighbors([query_features], n_neighbors=n_neighbors)
    
    similar_images = feature_vectors_df.iloc[indices[0]]['filename'].tolist()
    return similar_images, distances[0]

# Main recommendation function for local images
def recommend_similar_images_local(query_image_path, knn_model_path, feature_vectors_path, n_neighbors=5):
    # Load the feature extraction model
    feature_extraction_model = load_feature_extraction_model()
    
    # Extract features from the query image
    query_features = extract_features_from_path(query_image_path, feature_extraction_model)
    if query_features is None:
        print("Failed to extract features from the query image.")
        return
    
    # Load the KNN model and feature vectors
    knn_model, feature_vectors_df = load_knn_model_and_features(knn_model_path, feature_vectors_path)
    
    # Find similar images
    similar_images, distances = find_similar_images(query_features, knn_model, feature_vectors_df, n_neighbors)
    
    # Display results
    print(f"Top {n_neighbors} similar images for '{query_image_path}':")
    for i, (image, distance) in enumerate(zip(similar_images, distances)):
        print(f"{i + 1}. {image} (Distance: {distance:.4f})")

# Example usage
if __name__ == "__main__":
    # Paths to the KNN model and feature vectors CSV
    knn_model_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/models/knn_model1.pkl'
    feature_vectors_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'
    
    # Path to the local query image
    query_image_path = "/home/abhaya/Desktop/tt.png"
    
    # Get recommendations
    recommend_similar_images_local(query_image_path, knn_model_path, feature_vectors_path, n_neighbors=5) 