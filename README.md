# Weather Classification Model

## a. Problem Statement
The objective of this project is to build a machine learning model capable of classifying weather conditions based on meteorological data. By analyzing features such as temperature, humidity, precipitation, and wind speed, the model aims to predict the severity or type of weather conditions. This classification helps in understanding weather patterns and can be used for various forecasting and planning applications.

## b. Dataset Description
Dataset Source : "https://www.kaggle.com/datasets/yug201/daily-climate-time-series-data-delhi-india"
The dataset used in this project contains weather information with the following characteristics:
- **Total Rows:** 3557
- **Total Features:** 15 (after processing)
- **Target Label:** `condition_severity` (Encoded from weather conditions)

### Features:
1. `tempmax`: Maximum temperature
2. `tempmin`: Minimum temperature
3. `temp`: Average temperature
4. `humidity`: Humidity percentage
5. `precip`: Precipitation amount
6. `precipprob`: Probability of precipitation
7. `precipcover`: Precipitation coverage
8. `windspeed`: Wind speed
9. `sealevelpressure`: Sea level pressure
10. `month`: Month of the observation
11. `dayofweek`: Day of the week
12. `weekofyear`: Week of the year
13. `temp_humidity`: Interaction between temperature and humidity
14. `temp_range`: Difference between max and min temperature
15. `heat_index`: Calculated heat index

## c. Models Used
Six different machine learning models were implemented and evaluated on the dataset. The following table compares their performance across various metrics:

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.7907 | 0.9364 | 0.7892 | 0.7907 | 0.7889 | 0.6837 |
| Decision Tree | 0.7781 | 0.9102 | 0.7754 | 0.7781 | 0.7762 | 0.6640 |
| kNN | 0.7978 | 0.9115 | 0.7924 | 0.7978 | 0.7943 | 0.6934 |
| Naive Bayes | 0.7374 | 0.9203 | 0.7554 | 0.7374 | 0.7387 | 0.6114 |
| Random Forest (Ensemble) | 0.8020 | 0.9466 | 0.7975 | 0.8020 | 0.7971 | 0.7003 |
| XGBoost (Ensemble) | 0.8132 | 0.9241 | 0.8111 | 0.8132 | 0.8117 | 0.7167 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
| :--- | :--- |
| Logistic Regression | Provides a solid baseline with good performance, especially when features are scaled. |
| Decision Tree | Captured non-linear patterns well but was slightly less accurate than the ensemble methods. |
| kNN | Showed strong results (Accuracy ~0.80) when distance-weighted neighbors were used, benefiting from the similarity in weather patterns. |
| Naive Bayes | While the fastest to train, it had the lowest accuracy, likely due to the independence assumption being violated by correlated weather features. |
| Random Forest (Ensemble) | Excellent performance with the highest AUC (0.9466), indicating very strong class separation capabilities. |
| XGBoost (Ensemble) | The best-performing model overall, achieving the highest Accuracy (0.8132) and F1 (0.8117), effectively handling complex feature interactions. |

### Plot
**Model Comparison: Accuracy**

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/e3d76009-3709-4b6e-8b23-7cc77cdce7f1" />

**Model Comparison: AUC**

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/b4d44fbc-e8ba-471e-a521-b6ae6c4d4b63" />

**Model Comparison: F1 Score**

<img width="630" height="470" alt="image" src="https://github.com/user-attachments/assets/b3be059f-4ab4-4700-a29e-f18bb5446aa5" />

