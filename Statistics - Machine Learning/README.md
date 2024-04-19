# Statistics - Machine Learning

## Description

This folder contains a series of Machine Learning exercises completed using R.

This was done as part of the course *MOD 2.3: Statistical Engineering*. The markdown elements of the notebooks are written in French. Due to the way these notebooks were imported, there may be issues regarding file paths when importing.

### 1- Linear Regression

**Objective**

Using data in file "ph.txt", use linear regression to model the relationship between the variables (temperature and the pH of the ocean) and time. Based on the fitted models, predict their values in the year 2050.

**Methodology**
- Utilize lm function in R to perform the linear regressions
- Visualize various plots to get a first impression on the precision of the model and how well it explains the variance:
    - **Regression Plot**: visually assess the linear relationship
    - **Residuals vs Fitted**: identify potential non-constant variance or non-linearity
    - **Q-Q Residuals**: evaluate if the residuals follow a normal distribution
    - **Scale-Location** and **Residuals vs Leverage**: detect outliers with undue influence on the model

- Conduct a Student's t-test to validate the model
- Calculate the 95% confidence interval for each prediction and conclude


### 2- Design of Experiments

**Objective**

Analyze the sensitivity of an exchange option price to various model parameters using a design of experiments.
Build statistical models to quantify the relationships between parameters and option price.
Evaluate the impact of parameter changes on the option price.

**Methodology**

- Import design of experiments from file __plan_experience_prix.txt__ and use __PX_exo1_finance.R__ script to calculate the option price for each experimental setting

- Propose a linear regression model
- Assess model quality using appropriate metrics
- Employ AIC to select a model with fewer parameters that explains the data effectively
- Present the adjusted model equation

### 3- Logistic Regression

**Objective**

Analyze the relationship between the binary response variable "Y" (likelihood of new cancer detection) and other explanatory variables in the "meexpV2.txt" dataset.
Identify statistically significant predictors of cancer risk.
Interpret the effects of different modalities of the influential variables on the response.
Evaluate the quality of the fitted logistic regression model.

**Methodology**

- Build a logistic regression model using glm function
- Determine the statistical influential variables and their effects on Y
- Evaluate the model
- Attempt to build a better model, incorporating interaction terms and AIC to determine significant parameters

### 4- Statistical Learning

#### Part 1: PLS, PCR, Lasso

**Objective**

Develop a predictive model to estimate the penetration of bitumen based on its infrared (IR) spectrum using Partial Least Squares Regression (PLSR), Principal Components Regression (PCR), and Lasso regression methods.

**Methodology**

- Use the pcr function from the pls package to fit a PCR model
    - Select the number of principal components using cross-validation
- Employ the plsr function from the pls package to fit a PLS model
    - Determine the optimal number of latent variables using cross-validation
- Use the lars package to fit a Lasso regression model


- Compare the predictive performance of the 3 models on the test set with RMSE

#### Part 2: CART, RF, Bagging, Boosting

**Objective**

Analyze the factors influencing baby car seat sales across 400 stores using various ensemble learning methods.
Compare the performance of CART (Classification and Regression Trees), Random Forest, Bagging, and Boosting algorithms.

**Methodology**

- Implement a CART model using the rpart package
- Build a Random Forest model using the randomForest package
- Implement a bagging algorithm using the ipred package
- Build a boosting model using the gbm package
- Evaluate the models using RMSE to compare accuracy
- Implement a backward stepwise selection procedure for a linear regression model.
