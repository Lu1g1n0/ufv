import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from pandas.plotting import scatter_matrix

''' Load the CSV file '''
#file_path = '/path/to/your/file.csv'
#data = pd.read_csv(file_path)
data = pd.read_csv('2.iris.csv')

sns.set(style="whitegrid")

# 1. Scatter Plot: Relationship between sepal_length and sepal_width
plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='sepal_length', y='sepal_width', hue='species', palette='viridis')
plt.title('Scatter Plot of Sepal Length vs. Sepal Width')
plt.xlabel('Sepal Length',fontsize=14)
plt.ylabel('Sepal Width',fontsize=14)
plt.legend(title='Species',fontsize=14)
plt.show()

# 2. Line Plot: Sequential visualization for petal_length
plt.figure(figsize=(8, 6))
sns.lineplot(data=data.reset_index(), x='index', y='petal_length', hue='species', palette='tab10')
plt.title('Line Plot of Petal Length')
plt.xlabel('Index',fontsize=14)
plt.ylabel('Petal Length',fontsize=14)
plt.legend(title='Species',fontsize=14)
plt.show()

# 3. Boxplot: Sepal Length grouped by species
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, x='species', y='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()

# 4. Histogram: Frequency distribution of petal_length
plt.figure(figsize=(8, 6))
sns.histplot(data=data, x='petal_length', bins=15, kde=True, color='skyblue')
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()

# 5. Spider Plot: Comparing average values of numerical features by species
# Compute averages for each species
averages = data.groupby('species').mean()
categories = averages.columns.tolist()

# Add a circular plot for each species
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
angles += angles[:1]
# Plot data for each species
for species in averages.index:
    values = averages.loc[species].tolist()
    values += values[:1]  # Close the circle
    ax.plot(angles, values, label=species)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Spider Plot of Average Measurements by Species')
plt.legend(title='Species', loc='upper right')
plt.show()

# 6. Bar Chart: Average petal_width by species
plt.figure(figsize=(8, 6))
sns.barplot(data=data, x='species', y='petal_width', ci=None, palette='muted')
plt.title('Bar Chart of Average Petal Width by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Width')
plt.show()

# 7. Heatmap: Correlation matrix of numeric columns
correlation_matrix = data.iloc[:, :-1].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Heatmap of Correlation Matrix')
plt.show()

# 8. Correlation Plot: Scatter matrix of numeric columns
scatter_matrix(data.iloc[:, :-1], figsize=(12, 12), alpha=0.8, diagonal='hist', color=['skyblue'])
plt.suptitle('Scatter Matrix of Numeric Features')
plt.show()
