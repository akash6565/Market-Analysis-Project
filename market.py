# Import necessary libraries
import pandas as pd

# Read market data from CSV file
market_data = pd.read_csv('smartphone_market_data.csv')

# Display top 5 rows of the dataset
print("Top 5 rows of the dataset:")
print(market_data.head())

# Analyze market trends
average_price = market_data['Price'].mean()
average_sales = market_data['Sales'].mean()

# Print results
print("\nAverage Price of Smartphones:", average_price)
print("Average Monthly Sales of Smartphones:", average_sales)
