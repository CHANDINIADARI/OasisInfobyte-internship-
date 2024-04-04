import pandas as pd

# Step 1: Data Loading
data = pd.read_csv("C:/project-3/AB_NYC_2019.csv")  

# Step 2: Data Exploration
print("First few rows of the dataset:")
print(data.head()) 
print("\nInformation about the dataset:")
print(data.info())  
print("\nSummary statistics:")
print(data.describe()) 

# Step 3: Missing Data Handling
print("\nHandling missing data:")
print("Number of missing values in each column:")
print(data.isnull().sum())  

# Example: Dropping rows with missing values 
# data.dropna(inplace=True)

# Step 4: Duplicate Removal
print("\nRemoving duplicates:")
initial_rows = len(data)
data.drop_duplicates(inplace=True)  
final_rows = len(data)
print(f"Number of duplicate records removed: {initial_rows - final_rows}")

# Step 5: Outlier Detection
print("\nOutlier detection:")