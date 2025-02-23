import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import requests
from io import BytesIO
from PIL import Image

def preprocess_image_from_url(url):
    if pd.isna(url) or not url.strip():
        return None
    
    try:
        response = requests.get(url, timeout=10)  
        response.raise_for_status()  
        img = Image.open(BytesIO(response.content))
        img = img.resize((224, 224))  
        img_array = np.array(img)

        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]  # Remove alpha channel if present

        img_array = img_array.astype('float32')  
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0, 1]
        print(f"Image shape: {img_array.shape}")  
        return img_array
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def extract_features_from_url(url, model):
    img_array = preprocess_image_from_url(url)
    if img_array is not None:
        # Ensure the input shape is correct
        if img_array.shape == (1, 224, 224, 3):
            features = model.predict(img_array)
            return features.flatten()
        else:
            print(f"Unexpected image shape: {img_array.shape}")
            return np.zeros((model.output_shape[1],))
    else:
        print(f"Invalid image from URL: {url}")
        return np.zeros((model.output_shape[1],))

def extract_all_features(df, model):
    result = []
    for _, row in df.iterrows():
        url = row.get('link')  
        filename = row.get('filename')  
        features = extract_features_from_url(url, model)
        result.append((filename, features))
    return result

def save_to_csv(filenames, feature_vectors, file_name='/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'):
    feature_df = pd.DataFrame(feature_vectors)
    feature_df['filename'] = filenames
    feature_df = feature_df[['filename'] + [col for col in feature_df.columns if col != 'filename']]
    feature_df.to_csv(file_name, index=False)
    print(f"Feature vectors have been saved to '{file_name}'.")

def main():

    model = ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
    print(f"Model output shape: {model.output_shape}")  

    df = pd.read_csv('/home/abhaya/Desktop/fafaFInal/myWork/src/data/images.csv')
    df = df.dropna(subset=['link', 'filename'])  
    
   
    result = extract_all_features(df, model)
    
  
    image_filenames, image_feature_vectors = zip(*result)
    
    
    save_to_csv(image_filenames, image_feature_vectors)

if __name__ == '__main__':
    main()