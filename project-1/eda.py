import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Data Loading and Cleaning
retail_data = pd.read_csv("retail_sales_dataset.csv")  
print(retail_data.head())  

# Check for missing values
print(retail_data.isnull().sum())

# Remove rows with missing values
retail_data.dropna(inplace=True)

# Step 2: Descriptive Statistics
# Calculate basic statistics
summary_stats = retail_data.describe()
print(summary_stats)

# Step 3: Time Series Analysis
# Convert the date column to datetime format
retail_data['Date'] = pd.to_datetime(retail_data['Date'])

# Group sales by date and sum the values
daily_sales = retail_data.groupby('Date')['Total Amount'].sum()

# Plotting sales trends over time
plt.figure(figsize=(10, 6))
plt.plot(daily_sales)
plt.title('Daily Sales Trends')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

# Step 4: Visualization
# Example: Bar chart of sales by product category
sales_by_category = retail_data.groupby('Product Category')['Total Amount'].sum()
sales_by_category.plot(kind='bar', figsize=(10, 6))
plt.title('Total Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.show()