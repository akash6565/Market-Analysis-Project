**Market Analysis Project**

**Objective:**
Conduct comprehensive market research to analyze trends in the smartphone industry.

**Description:**
This project aims to explore datasets containing information on smartphone models, brands, prices, and sales figures. Through statistical analysis and visualization techniques, key insights are extracted to understand market dynamics, consumer preferences, and competitive landscapes.

**Python Code (market_analysis.py):**
```python
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
```

**Output Screenshot:**

![Market Analysis Output](video_slides/source/vq_ma_3b.Rmd)

**How to Use:**
1. Ensure Python is installed on your system.
2. Download the 'smartphone_market_data.csv' file.
3. Run the 'market_analysis.py' script.
4. Review the output for insights into smartphone market trends.

**Dependencies:**
- Python 3.x
- pandas

**Contributing:**
- Fork the repository
- Make your changes
- Submit a pull request

**License:**
This project is licensed under the MIT License.

**Author:**
[AKASH PATI]

**Contact:**
[akhs3456@gmail.com]

**Acknowledgements:**
Special thanks to [skanz] for providing the dataset.

