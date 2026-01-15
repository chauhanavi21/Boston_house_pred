# Boston House Price Prediction using Machine Learning

## Project Overview

This repository contains a predictive analytics solution designed to estimate property values in the Boston metropolitan area. Leveraging the renowned Boston Housing dataset, the implementation constructs an intelligent pricing model by examining key property characteristics including neighborhood safety metrics, dwelling size, transportation accessibility, and various socio-economic indicators. The solution employs CatBoostRegressor, a gradient boosting framework, to deliver precise price forecasts. Throughout the development process, comprehensive data exploration, feature engineering, and model optimization techniques were applied. The final model serves as a practical decision-support tool for real estate stakeholders seeking data-driven price estimations.

## Technology Stack

This project is built using the following technologies and libraries:

- **Flask** - Web framework for application deployment
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing operations
- **matplotlib** - Data visualization
- **scikit-learn** - Machine learning utilities and preprocessing
- **catboost** - Gradient boosting regression algorithm
- **gunicorn** - Production WSGI HTTP server

## Installation & Setup

Follow these instructions to set up and run the project on your local machine:

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd House-Price-Prediction
```

### Step 2: Install Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Step 3: Generate Model Files

Execute the Jupyter notebook (`Boston House Price Prediction.ipynb`) to train the model and generate the necessary `.pkl` files required for deployment. Once the notebook runs successfully, you'll have both the trained model (`housepred.pkl`) and the scaler (`scaler.pkl`) ready for the web application.

## Dataset Information

### Boston Housing Dataset

The dataset originates from the `sklearn.datasets` library and comprises 506 property records from the Boston area. Each record contains 13 quantitative attributes that characterize different aspects of residential properties. The dataset's target variable represents the median home value in thousands of dollars for owner-occupied residences.

### Feature Description

The model analyzes the following 13 input features:

1. **CRIM** - Crime rate per capita within the town
2. **ZN** - Percentage of residential land allocated for lots exceeding 25,000 square feet
3. **INDUS** - Percentage of non-retail commercial acreage per town
4. **CHAS** - Binary indicator for Charles River proximity (1 = adjacent to river, 0 = otherwise)
5. **NOX** - Nitric oxides concentration measured in parts per 10 million
6. **RM** - Mean number of rooms per residential unit
7. **AGE** - Percentage of owner-occupied homes constructed before 1940
8. **DIS** - Weighted distance measurements to five major Boston employment hubs
9. **RAD** - Accessibility index for radial highway systems
10. **TAX** - Property tax rate per $10,000 of full assessed value
11. **PTRATIO** - Average student-to-teacher ratio within the town's schools
12. **B** - Calculated as 1000(Bk - 0.63)² where Bk represents the proportion of African American residents by town
13. **LSTAT** - Percentage of residents classified as lower socioeconomic status

### Target Variable

- **MEDV** - Median home value expressed in thousands of dollars ($1000s)

## Data Preprocessing Pipeline

Prior to model training, the raw dataset undergoes systematic preprocessing. The workflow begins by extracting the feature matrix (X) and target vector (y) from the dataset. Feature normalization is accomplished using scikit-learn's `StandardScaler` transformer, which standardizes all input variables to have zero mean and unit variance. This scaling ensures that features with larger magnitudes don't dominate the learning process. The dataset is subsequently partitioned into training and validation subsets following an 80-20 split convention.

## Model Development & Performance Assessment

The predictive model is constructed using the CatBoostRegressor framework. Model development involves several critical steps: feature preparation, algorithm selection, and hyperparameter optimization. A RandomizedSearchCV strategy is implemented to systematically explore the hyperparameter space, with optimal configuration selection guided by 5-fold cross-validation performance. The final model is fitted on the training partition using the best-performing hyperparameter combination. Model validation occurs on the held-out test set, where predictions are benchmarked against actual home values using the R² (coefficient of determination) evaluation metric.

**Primary Algorithm**: CatBoost Regression

## Web Application Deployment

The project features a Flask web service that enables interactive price prediction. Upon server initialization, the application deserializes the trained model from `housepred.pkl` and the preprocessing scaler from `scaler.pkl`. Users can submit property feature values via two interfaces:

- **Web Form Interface**: HTML form for direct user interaction
- **REST API Endpoint**: JSON-based API (`/predict_api`) for programmatic access

Both interfaces accept the 13 feature inputs, apply appropriate preprocessing, and return the estimated house price prediction.

## How to Contribute

The open-source ecosystem thrives on collaborative innovation. Your contributions, whether through bug reports, feature suggestions, or code improvements, are warmly welcomed and highly valued.

To contribute:

1. Fork this repository to your GitHub account
2. Create a new branch for your feature or fix
3. Make your modifications and test thoroughly
4. Commit your changes with clear, descriptive messages
5. Push your branch and initiate a pull request

For enhancement ideas, feel free to open an issue tagged with "enhancement". If you find this project useful, consider giving it a star!

## License

This project is licensed under the GNU General Public License v3.0. For complete license details, refer to the `LICENSE` file.

## Credits & References

The foundation of this work stems from the Boston House Price Prediction challenges and datasets available on Kaggle. We extend gratitude to the developers and maintainers of the open-source Python libraries that made this project possible.
