import pandas as pd
import numpy as np

def remove_zero_vectors(csv_file_path):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Identify rows where the feature vector is all zeros
    # Assuming the feature vectors start from the second column
    feature_columns = df.columns[1:]  # Exclude the 'filename' column
    zero_vector_mask = (df[feature_columns] == 0).all(axis=1)
    
    # Remove rows where the feature vector is all zeros
    df_cleaned = df[~zero_vector_mask]
    
    # Save the updated DataFrame back to the same CSV file
    df_cleaned.to_csv(csv_file_path, index=False)
    print(f"Cleaned data saved to '{csv_file_path}'.")

# Path to the CSV file
csv_file_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'

# Remove rows with all-zero feature vectors
remove_zero_vectors(csv_file_path)