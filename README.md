# ridge_regression_forecast
# Jeep Wrangler Sales Prediction using Ridge Regression

## Project Summary
This project utilizes historical sales data from 2010 to 2018 to develop a ridge regression model for predicting Jeep Wrangler's monthly sales. The model incorporates economic and search-related indicators such as the unemployment rate, Wrangler-related Google search queries, and Consumer Price Index (CPI) indices.

By implementing ridge regression, we introduce regularization to prevent overfitting and enhance the model’s generalizability. After tuning the regularization parameter (lambda, \( \lambda \)), the optimized model is used to predict future Wrangler sales based on new feature values.

## Why This Project is Important
Accurate sales prediction is crucial for automobile manufacturers and dealers to make data-driven decisions regarding production, marketing, and resource allocation. Jeep can leverage this model to:
- Optimize inventory management.
- Predict market demand fluctuations.
- Plan marketing strategies based on economic indicators and consumer interest.

This project demonstrates how machine learning techniques can be applied to real-world business challenges and improve decision-making.

## Tools & Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn (sklearn), Matplotlib
- **Machine Learning Model:** Ridge Regression
- **Performance Metric:** Root Mean Squared Error (RMSE)

## Methodology
### 1. Data Loading and Exploration
The dataset is loaded and viewed to see the structure of the data. On inspection, it contains monthly sales data along with key economic indicators.

### 2. Feature Selection
To build the model, relevant features are selected based on their relationship with Wrangler sales. The chosen features include:
- **Year:** Captures overall market trends.
- **Unemployment Rate:** Reflects economic conditions affecting consumer purchasing power.
- **Wrangler Queries:** Google search queries related to Jeep Wrangler, indicating consumer interest.
- **CPI Energy & CPI All:** Measures inflation trends in general and energy-specific markets.

```

### 3. Data Splitting
The data is split into training (2010-2017) and test sets (2018) to evaluate model performance.

```python
# Split data into training (2010-2017) and test (2018) sets
train_data = data[data['Year'] < 2018]
test_data = data[data['Year'] == 2018]

X_train = train_data[['Year', 'Unemployment.Rate', 'Wrangler.Queries', 'CPI.Energy', 'CPI.All']]
y_train = train_data['Wrangler.Sales']

X_test = test_data[['Year', 'Unemployment.Rate', 'Wrangler.Queries', 'CPI.Energy', 'CPI.All']]
y_test = test_data['Wrangler.Sales']
```

### 4. Ridge Regression and Hyperparameter Tuning
Ridge regression is used instead of ordinary least squares regression to mitigate overfitting by adding a penalty term controlled by the regularization parameter \( \lambda \). The model is trained for multiple \( \lambda \) values to find the optimal one that minimizes RMSE. Ridge Regression is a type of linear regression that includes a regularization term to prevent overfitting. It modifies the ordinary least squares (OLS) regression by adding a penalty term to the loss function that shrinks the regression coefficients. This penalty is controlled by a hyperparameter, lambda (λ).

There are other types of machine learning methods and data analysis methdods that also might have been suited for this task. 
If interpretability is important → Ridge Regression or Lasso (understandable coefficients).
If feature selection is needed → Lasso or Elastic Net.
If we expect non-linear relationships → Decision Trees, Random Forest, or XGBoost.
If we have very large data → Neural Networks or Gradient Boosting.


### 5. Model Performance Visualization
To visualize the impact of different \( \lambda \) values, RMSE is plotted against \( \lambda \).


The best \( \lambda \) value is found to be **390**, which minimizes RMSE.

### 6. Final Model Training and Prediction
After selecting the optimal \( \lambda \), the final Ridge regression model is trained and used to predict Jeep Wrangler’s 2018 sales.


The model predicts **16,993 sales** for the Wrangler in 2018.

## Conclusion
This project illustrates how ridge regression can be used to predict vehicle sales by leveraging economic and search-related features. The findings can assist Jeep and similar companies in making data-driven business decisions. By incorporating regularization, the model achieves a balance between predictive power and generalization, ensuring reliable forecasts for future sales trends.

### Future Improvements
- **Feature Engineering:** Incorporate additional macroeconomic indicators, social media trends, and seasonal effects.
- **Time Series Modeling:** Implement advanced models like ARIMA, LSTM, or XGBoost to capture temporal dependencies.
- **Hyperparameter Optimization:** Use grid search or Bayesian optimization to fine-tune \( \lambda \) more efficiently.

This project highlights the power of machine learning in real-world applications, demonstrating how predictive modeling can enhance decision-making in the automotive industry.



