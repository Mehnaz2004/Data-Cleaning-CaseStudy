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

# Total Laid Off - Before
plt.subplot(1, 2, 1)
sns.distplot(dataset['total_laid_off'])
plt.title("Before")

# Remove outliers from total_laid_off using z_score
z_score = np.abs(stats.zscore(dataset['total_laid_off']))  # computing z_scores
dataset = dataset[(z_score < 3)]

# Total Laid Off - After
plt.subplot(1, 2, 2)
sns.distplot(dataset['total_laid_off'])
plt.title("After")
plt.show()


# Funds Raised Millions - Before
plt.subplot(1, 2, 1)
sns.distplot(dataset['funds_raised_millions'])
plt.title("Before")

# IQR method for funds_raised_millions
IQR = dataset['funds_raised_millions'].quantile(0.75) - dataset['funds_raised_millions'].quantile(0.25)
min_range = dataset['funds_raised_millions'].quantile(0.25) - (1.5 * IQR)
max_range = dataset['funds_raised_millions'].quantile(0.75) + (1.5 * IQR)
dataset = dataset[dataset['funds_raised_millions'] <= max_range]

# Funds Raised Millions - After
plt.subplot(1, 2, 2)
sns.distplot(dataset['funds_raised_millions'])
plt.title("After")
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

# Boxplot for total_laid_off - Before
plt.subplot(1, 2, 1)
sns.boxplot(x=dataset['total_laid_off'])
plt.title("Before")
dataset['total_laid_off'] = min_max_scaler.fit_transform(dataset[['total_laid_off']])
plt.subplot(1, 2, 2)
sns.boxplot(x=dataset['total_laid_off'])
plt.title("After")
plt.show()

# Boxplot for funds_raised_millions - Before
plt.subplot(1, 2, 1)
sns.boxplot(x=dataset['funds_raised_millions'])
plt.title("Before")
dataset['funds_raised_millions'] = min_max_scaler.fit_transform(dataset[['funds_raised_millions']])
plt.subplot(1, 2, 2)
sns.boxplot(x=dataset['funds_raised_millions'])
plt.title("After")
plt.show()

print(dataset.describe())
print(dataset.shape)
print("Cleaned data now saved to a file.")

dataset.to_csv('cleaned_data.csv', index=False)

