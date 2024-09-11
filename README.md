# Data Cleaning Case Study

## Overview

This repository contains an exercise on data cleaning using a dataset focused on company layoffs. The aim was to apply data cleaning techniques to prepare the dataset for further analysis, visualization, and modeling. This exercise helps demonstrate the importance of data cleaning in ensuring data integrity and reliability for effective data analysis and machine learning.

## Dataset

The dataset used for this exercise is `layoffs.csv`, which includes various attributes related to layoffs in different companies.

## Libraries Used

- **Pandas**: For data manipulation and analysis.
- **Seaborn**: For creating informative and attractive visualizations.
- **Matplotlib**: For additional plotting capabilities.
- **Scipy**: For statistical functions.
- **Scikit-learn**: For preprocessing tasks.

## Steps Undertaken

1. **Exploration:**
   - Loaded the dataset and performed initial exploration to identify missing values and get summary statistics.

2. **Handling Missing Values:**
   - Addressed missing values by using mode for categorical data and mean/median for numerical fields based on their distribution.

3. **Outlier Detection and Reduction:**
   - Detected outliers using z-score for `total_laid_off` and IQR for `funds_raised_millions`.
   - Utilized boxplots and distplots to visualize the impact of outliers and the effectiveness of the chosen methods.

4. **Data Type Conversion:**
   - Converted the date column to datetime type to ensure proper formatting.

5. **Removing Duplicates:**
   - Cleaned up duplicate entries to enhance data quality.

6. **Encoding & Scaling:**
   - Applied label encoding to categorical columns and Min-Max scaling to numerical features.

## Key Visualizations

- **Boxplots and Distplots**: Visualized the distribution and characteristics of the data to aid in understanding and handling various data integrity issues.

## Conclusion

This exercise provided hands-on experience with data cleaning techniques, illustrating their importance in preparing data for accurate analysis and modeling. The process involved trial and error to find the most effective methods, making the learning experience both comprehensive and engaging.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Mehnaz2004/Data-Cleaning-CaseStudy.git
2. Navigate to the Directory:
   ```bash
   cd Data-Cleaning-CaseStudy
3. Run the Data Cleaning Script:
   ```bash
   python data_cleaning_script.py

   
