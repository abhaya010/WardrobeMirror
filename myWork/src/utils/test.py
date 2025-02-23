import pandas as pd

# Load the CSV file
file_path = '/home/abhaya/Desktop/fafaFInal/myWork/src/data/image_feature_vectors.csv'
df = pd.read_csv(file_path)

# Print the first few rows of the DataFrame
print(df.head())