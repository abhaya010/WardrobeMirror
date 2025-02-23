import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tensorflow.keras.applications import ResNet50
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

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

# Load the KNN model, feature vectors, and images.csv
def load_knn_model_and_features(knn_model_path, feature_vectors_path, images_csv_path):
    knn_model = joblib.load(knn_model_path)
    feature_vectors_df = pd.read_csv(feature_vectors_path)
    images_df = pd.read_csv(images_csv_path)  # Load the images.csv file
    return knn_model, feature_vectors_df, images_df

# Find similar images using the KNN model
def find_similar_images(query_features, knn_model, feature_vectors_df, images_df, n_neighbors=5):
    X = feature_vectors_df.drop(columns=['filename']).values  # Convert to numpy array
    distances, indices = knn_model.kneighbors([query_features], n_neighbors=n_neighbors)
    
    similar_filenames = feature_vectors_df.iloc[indices[0]]['filename'].tolist()
    
    # Get the URLs of the similar images from images.csv
    similar_images = images_df[images_df['filename'].isin(similar_filenames)]['link'].tolist()
    
    return similar_images, distances[0]

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the uploaded file temporarily
    upload_folder = 'uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    
    # Load the feature extraction model
    feature_extraction_model = load_feature_extraction_model()
    
    # Extract features from the query image
    query_features = extract_features_from_path(file_path, feature_extraction_model)
    if query_features is None:
        return jsonify({"error": "Failed to extract features from the query image"}), 400
    
    # Load the KNN model, feature vectors, and images.csv
    knn_model_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/models/knn_model1.pkl'
    feature_vectors_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'
    images_csv_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/images.csv'  # Path to images.csv
    knn_model, feature_vectors_df, images_df = load_knn_model_and_features(knn_model_path, feature_vectors_path, images_csv_path)
    
    # Find similar images
    similar_images, distances = find_similar_images(query_features, knn_model, feature_vectors_df, images_df, n_neighbors=5)
    
    # Prepare the response
    response = {
        "query_image": f"/uploads/{file.filename}",  # Path to the query image
        "similar_images": similar_images,  # List of URLs to similar images
        "distances": distances.tolist()
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5500, debug=False)  # Disable debug mode