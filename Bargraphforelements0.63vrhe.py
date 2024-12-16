# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 17:36:55 2024

@author: merye
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel data
file_path = r"C:\Users\merye\OneDrive\Masaüstü\Tez1\0.63VRHE_data_after_formulas_updated.xlsx"  # Replace with your actual file path
df = pd.read_excel(file_path)


# Define a function to count the number of elements in the formula
def count_elements(formula):
    # Split the formula by elements (A, B, C, D, etc.) and count the unique elements
    elements = ['Mn', 'Ni', 'Ca', 'Fe', 'La', 'Y', 'In']
    count = sum([element in formula for element in elements])
    return count
# Apply the function to the 'formula' column
df['num_elements'] = df['formula'].apply(count_elements)

# Filter rows where the number of elements is between 1 and 4
filtered_df = df[df['num_elements'].isin([1, 2, 3, 4])]

# Count the occurrences of each formula type based on the number of elements
element_counts = filtered_df['num_elements'].value_counts().sort_index()

# Plot the data
plt.figure(figsize=(8, 6))

# Create a bar plot for the count of formulas with 1, 2, 3, and 4 elements
ax = element_counts.plot(kind='bar', color='purple')

# Add labels to the bars
for i, count in enumerate(element_counts):
    ax.text(i, count + 0.05, str(count), ha='center', va='bottom', fontsize=12)

# Add labels and title
plt.xlabel('Number of Metal Elements in Formula')
plt.ylabel('Count of Formulas')
plt.title('0.63 V_RHE')
plt.xticks(rotation=0)
plt.grid(True)

# Show the plot
plt.show()
