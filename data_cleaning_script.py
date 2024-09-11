import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Loading the data into a DataFrame using pandas
dataset = pd.read_csv(r"layoffs.csv")

# Exploring the data
print("The percentage of null values overall:\n", (dataset.isnull().sum().sum() / (dataset.shape[0] * dataset.shape[1])) * 100)
print("View the first few rows\n", dataset.head())  # View the first few rows
print("Get information about data types and missing values\n", dataset.info())  # Get information about data types and missing values
print("Get summary statistics\n", dataset.describe())  # Get summary statistics

# HANDLING MISSING VALUES
print("Null Values per Column before handling them:\n", (dataset.isnull().sum() / dataset.shape[0]) * 100)  # none are above 50% so we can choose to fill them using mean mode or other methods

# Filling in categorical columns
dataset['industry'] = dataset['industry'].fillna(dataset['industry'].mode()[0])
dataset['stage'] = dataset['stage'].fillna(dataset['stage'].mode()[0])

# Forward filling the dates
dataset['date'] = dataset['date'].fillna(method='ffill')

# Filling in numerical columns
dataset['percentage_laid_off'] = dataset['percentage_laid_off'].fillna(dataset['percentage_laid_off'].mean())  # it is normally distributed with little chances of outliers so we use mean
dataset['total_laid_off'] = dataset['total_laid_off'].fillna(dataset['total_laid_off'].median())  # it is not normally distributed and thus used median
dataset['funds_raised_millions'] = dataset['funds_raised_millions'].fillna(dataset['funds_raised_millions'].median())  # it is not normally distributed and thus used median

print("Null Values per Column After handling them:\n", (dataset.isnull().sum() / dataset.shape[0]) * 100)

# REPLACING AND HANDLING DATATYPES
dataset['date'] = pd.to_datetime(dataset['date'])  # Convert the date column to datetime type

# OUTLIER DETECTION
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Total Laid Off - Before
sns.histplot(dataset['total_laid_off'], kde=True, color='skyblue', ax=axs[0, 0])
axs[0, 0].set_title('Total Laid Off - Before Outlier Removal')

# Funds Raised Millions - Before
sns.histplot(dataset['funds_raised_millions'], kde=True, color='salmon', ax=axs[0, 1])
axs[0, 1].set_title('Funds Raised (Millions) - Before Outlier Removal')

# Remove outliers from total_laid_off
z_score = np.abs(stats.zscore(dataset['total_laid_off']))  # computing z_scores
dataset_no_outliers_total = dataset[(z_score < 3)]

# Total Laid Off - After
sns.histplot(dataset_no_outliers_total['total_laid_off'], kde=True, color='skyblue', ax=axs[1, 0])
axs[1, 0].set_title('Total Laid Off - After Outlier Removal')

# IQR method for funds_raised_millions
IQR = dataset['funds_raised_millions'].quantile(0.75) - dataset['funds_raised_millions'].quantile(0.25)
min_range = dataset['funds_raised_millions'].quantile(0.25) - (1.5 * IQR)
max_range = dataset['funds_raised_millions'].quantile(0.75) + (1.5 * IQR)
dataset_no_outliers_funds = dataset[dataset['funds_raised_millions'] <= max_range]

# Funds Raised Millions - After
sns.histplot(dataset_no_outliers_funds['funds_raised_millions'], kde=True, color='salmon', ax=axs[1, 1])
axs[1, 1].set_title('Funds Raised (Millions) - After Outlier Removal')

plt.tight_layout()
plt.show()

# HANDLING DUPLICATES
dataset.drop_duplicates(inplace=True)

# ENCODING THE DATASET
# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Apply Label Encoding to categorical columns
categorical_columns = ['company', 'location', 'industry', 'country', 'stage']
for column in categorical_columns:
    dataset[column] = label_encoder.fit_transform(dataset[column])
    
# FEATURE SCALING

# Initialize MinMaxScaler
min_max_scaler = MinMaxScaler()

# List of columns to scale (excluding date and the target variable if applicable)
columns_to_scale = ['total_laid_off', 'percentage_laid_off', 'funds_raised_millions']

# Plotting feature scaling using boxplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot for total_laid_off - Before
sns.boxplot(x=dataset['total_laid_off'], ax=axs[0, 0], color='lightblue')
axs[0, 0].set_title('Total Laid Off - Before Scaling')

# Boxplot for funds_raised_millions - Before
sns.boxplot(x=dataset['funds_raised_millions'], ax=axs[0, 1], color='lightcoral')
axs[0, 1].set_title('Funds Raised (Millions) - Before Scaling')

# Apply MinMaxScaler
dataset[columns_to_scale] = min_max_scaler.fit_transform(dataset[columns_to_scale])

# Boxplot for total_laid_off - After
sns.boxplot(x=dataset['total_laid_off'], ax=axs[1, 0], color='lightblue')
axs[1, 0].set_title('Total Laid Off - After Scaling')

# Boxplot for funds_raised_millions - After
sns.boxplot(x=dataset['funds_raised_millions'], ax=axs[1, 1], color='lightcoral')
axs[1, 1].set_title('Funds Raised (Millions) - After Scaling')

plt.tight_layout()
plt.show()

print(dataset.describe())
print(dataset.shape)
print("Cleaned data now saved to a file.")

dataset.to_csv('cleaned_data.csv', index=False)

