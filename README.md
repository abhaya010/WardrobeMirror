# Fashion Recommendation System

## Overview

This project is a content-based fashion recommendation system that suggests outfits and clothing items based on an uploaded image. The system extracts features from the uploaded image, finds similar items using a KNN model, and provides personalized fashion recommendations.

## Project Structure

The project is divided into the following components:

- **Machine Learning Model (Flask Server)** – Extracts image features, performs recommendations.
- **Backend (Flask Server)** – Handles API requests, communicates with the machine learning model.
- **Frontend (HTML/CSS/JavaScript)** – Provides the user interface for uploading images and displaying recommendations.
- **Database (CSV Files)** – Stores feature vectors and metadata of fashion items.

## Installation and Setup

### 1. Setting Up the Machine Learning Model

Navigate to the `src` directory and ensure:

- A virtual environment is activated.
- Python version is **3.11 or below**.
- All required libraries are installed using:
  pip install -r requirements.txt
  

#### Preprocessing the Dataset

Run the following Python scripts in order to preprocess the dataset:

python merge_csv.py
python clean.py
python encoding.py
python tfidf.py
python normalize.py
```

Once preprocessing is complete, the dataset is ready for training.

#### Training the Model

Run the following command to train the KNN model:


python models/train_knn.py


#### Running the Flask Server

Once the model is trained, start the Flask server:


python src/main.py


The Flask server runs on **port 5500** and handles file uploads and recommendations.

### 2. Setting Up the Frontend

The frontend is a simple HTML/CSS/JavaScript interface. Ensure that the `templates` and `static` directories are correctly set up.

## Environment Variables

The project requires environment variables to be set up inside the `src` directory. These environment variables should be configured according to your system and credentials before running the project.

## Running the Application

1. Install dependencies in the ML model (`pip install -r requirements.txt`).
2. Start the **Flask server** (`python src/main.py` on port **5500**).
3. Open the web application in your browser, upload an image, and receive fashion recommendations.

## Workflow

1. The user uploads an image via the frontend.
2. The file is sent to the Flask backend via API calls.
3. Flask temporarily saves the file and extracts features using a pre-trained model.
4. Flask finds similar fashion items using KNN and returns item indices and metadata.
5. The recommendations are sent back to the frontend and displayed to the user.

## Notes

- Ensure that all dependencies are installed before running the project.
- The environment variables must be correctly set up inside the `src` directory.
- Flask runs on **port 5500**.

Enjoy using the fashion recommendation system!