import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from pandas.plotting import scatter_matrix

''' Load the CSV file '''
#file_path = '/path/to/your/file.csv'
#data = pd.read_csv(file_path)
data = pd.read_csv('2.iris.csv')


'''Outlier for a variable'''
# Step 1: Inject Outliers into the 'sepal_length' column
data_with_outliers = data.copy()
outlier_values = [15, 16, 17]  # Extreme values as outliers
data_with_outliers.loc[len(data_with_outliers)] = [15, 3, 1.5, 0.2, 'setosa']  # Add one outlier
data_with_outliers.loc[len(data_with_outliers)] = [16, 2.5, 1.4, 0.2, 'setosa']  # Add another
data_with_outliers.loc[len(data_with_outliers)] = [17, 3.5, 1.6, 0.3, 'setosa']  # Another one

# Step 2: Visualize Outliers using Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_with_outliers, x='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length with Outliers')
plt.xlabel('Sepal Length')
plt.show()

# Step 3: Visualize Outliers using Histogram
plt.figure(figsize=(8, 6))
sns.histplot(data=data_with_outliers, x='sepal_length', bins=20, kde=True, color='skyblue')
plt.title('Histogram of Sepal Length with Outliers')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()


'''Normal value for a variable but outlier if considering the specific class'''
# Step 1: Inject Class-Specific Outliers into the 'sepal_length' column
data_with_class_outliers = data.copy()

# Injecting values for 'setosa' that are normal for other species
setosa_outliers = [7.5, 8.0, 7.8]  # Values normal for other species but outliers for 'setosa'
for value in setosa_outliers:
    data_with_class_outliers.loc[len(data_with_class_outliers)] = [value, 3.2, 1.0, 0.2, 'setosa']

# Step 1: Boxplot Without Dividing by Class
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_with_class_outliers, x='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length (No Class Division)')
plt.xlabel('Sepal Length')
plt.show()

# Step 2: Histogram Without Dividing by Class
plt.figure(figsize=(8, 6))
sns.histplot(data=data_with_class_outliers, x='sepal_length', bins=20, kde=True, color='skyblue')
plt.title('Histogram of Sepal Length (No Class Division)')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()

# Step 2: Visualize using Boxplot (Grouped by species)
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_with_class_outliers, x='species', y='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length with Class-Specific Outliers')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()

# Step 3: Visualize using Histogram (Colored by species)
plt.figure(figsize=(8, 6))
sns.histplot(data=data_with_class_outliers, x='sepal_length', hue='species', kde=True, palette='viridis', bins=20)
plt.title('Histogram of Sepal Length with Class-Specific Outliers')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()


'''Normal value in a variable but outlier if associated with another variable'''
# Step 1: Inject a value that is normal for 'sepal_length' but an outlier for 'sepal_width'
data_with_relationship_outliers = data.copy()

# Inject the normal 'sepal_length' (e.g., 5.0) with an outlier 'sepal_width' (e.g., 1.0)
outlier_point = pd.DataFrame({'sepal_length': [5.0], 'sepal_width': [1.0], 'petal_length': [1.5], 'petal_width': [0.2], 'species': ['setosa']})
data_with_relationship_outliers = pd.concat([data_with_relationship_outliers, outlier_point], ignore_index=True)

# Step 2: Scatter Plot to Visualize the Relationship with Outlier Colored Differently
plt.figure(figsize=(8, 6))

# Plot the normal points
sns.scatterplot(data=data_with_relationship_outliers, x='sepal_length', y='sepal_width', color='skyblue')

# Highlight the outlier in red
sns.scatterplot(data=outlier_point, x='sepal_length', y='sepal_width', color='red', s=100, label='Outlier')

# Titles and labels
plt.title('Scatter Plot of Sepal Length vs. Sepal Width with Outlier Highlighted')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Show the plot
plt.legend()
plt.show()

'''Imputation'''
# Step 1: Compute the Mean of each column
mean_sepal_length = data['sepal_length'].mean()
mean_sepal_width = data['sepal_width'].mean()

# Step 2: Inject Outliers (same as before)
data_with_imputation_outliers = data.copy()

# Outlier points to inject
outlier_points = pd.DataFrame({
    'sepal_length': [4.8, 5.2, 5.5],  # Normal sepal_length values
    'sepal_width': [0.5, 1.5, 4.5],   # Extreme sepal_width values
    'petal_length': [1.4, 1.6, 1.5],
    'petal_width': [0.2, 0.3, 0.2],
    'species': ['setosa', 'versicolor', 'virginica']
})

# Add outliers to the dataset
data_with_imputation_outliers = pd.concat([data_with_imputation_outliers, outlier_points], ignore_index=True)

# Step 3: Before Imputation - Plot boxplots for sepal_length and sepal_width
plt.figure(figsize=(10, 6))

# Boxplot for sepal_length (Before imputation)
plt.subplot(1, 2, 1)
sns.boxplot(data=data_with_imputation_outliers, y='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length Before Imputation')
plt.ylabel('Sepal Length')

# Boxplot for sepal_width (Before imputation)
plt.subplot(1, 2, 2)
sns.boxplot(data=data_with_imputation_outliers, y='sepal_width', palette='pastel')
plt.title('Boxplot of Sepal Width Before Imputation')
plt.ylabel('Sepal Width')

plt.tight_layout()
plt.show()

# Step 4: Impute Outliers with the Mean Values
# Replace outliers' values with the mean values
data_with_imputation_outliers.loc[data_with_imputation_outliers['sepal_length'].isin([4.8, 5.2, 5.5]), 'sepal_length'] = mean_sepal_length
data_with_imputation_outliers.loc[data_with_imputation_outliers['sepal_width'].isin([0.5, 1.5, 4.5]), 'sepal_width'] = mean_sepal_width

# Step 5: After Imputation - Plot boxplots for sepal_length and sepal_width
plt.figure(figsize=(10, 6))

# Boxplot for sepal_length (After imputation)
plt.subplot(1, 2, 1)
sns.boxplot(data=data_with_imputation_outliers, y='sepal_length', palette='pastel')
plt.title('Boxplot of Sepal Length After Imputation')
plt.ylabel('Sepal Length')

# Boxplot for sepal_width (After imputation)
plt.subplot(1, 2, 2)
sns.boxplot(data=data_with_imputation_outliers, y='sepal_width', palette='pastel')
plt.title('Boxplot of Sepal Width After Imputation')
plt.ylabel('Sepal Width')

plt.tight_layout()
plt.show()

