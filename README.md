# churnprediction
This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The objective was to identify customers likely to discontinue services and evaluate different modeling and data balancing strategies to achieve optimal performance.

1. Data Loading and Initial Analysis

The dataset WA_Fn-UseC_-Telco-Customer-Churn.csv was loaded into a pandas DataFrame for analysis. Initial exploration included examining the dataset’s shape, previewing sample records, and reviewing data types and unique values across all features. Missing values in the TotalCharges column were identified and addressed by replacing empty strings with 0.0 and converting the column to a floating-point data type.

2. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand data distributions and relationships. Numerical features such as tenure, MonthlyCharges, and TotalCharges were analyzed using histograms and box plots. A correlation heatmap was used to examine relationships among numerical variables. Categorical features were analyzed using count plots to visualize their frequency distributions.

3. Data Preprocessing

The target variable Churn was encoded into numerical format, with Yes mapped to 1 and No mapped to 0. All remaining categorical features were converted into numerical values using Label Encoding. The trained label encoders were saved using pickle to ensure consistent preprocessing during future inference.

4. Initial Model Training and Evaluation

The dataset was split into training and testing sets. Due to class imbalance in the target variable, SMOTE (Synthetic Minority Over-sampling Technique) was applied to the training data to create a balanced dataset. Three classification models—Decision Tree, Random Forest, and XGBoost—were trained on the SMOTE-resampled data. Model performance was evaluated using 5-fold cross-validation, including Stratified K-Fold, where Random Forest and XGBoost demonstrated superior performance compared to the Decision Tree model.

5. Hyperparameter Tuning

To further improve model performance, GridSearchCV with Stratified K-Fold cross-validation was used to tune hyperparameters for both Random Forest and XGBoost models. The tuning process identified optimal parameter configurations that resulted in improved and more stable accuracy scores during cross-validation.

6. Alternative Data Balancing Strategy

In addition to SMOTE, a manual downsampling approach was implemented on the original training data. The majority class was randomly reduced to match the minority class, creating a balanced training dataset for comparison.

7. Evaluation on Downsampled Data

The tuned Random Forest and XGBoost models were evaluated on the downsampled dataset using Stratified K-Fold cross-validation. Results showed significantly lower performance compared to models trained on SMOTE-resampled data, indicating that downsampling was less effective for this problem.

8. Model Selection and Final Evaluation

Based on comparative analysis across different models, tuning stages, and data balancing techniques, SMOTE was identified as the more effective balancing strategy. Among all models, the hyperparameter-tuned Random Forest trained on SMOTE-resampled data consistently delivered the best performance. This model was selected as the final model and retrained on the full SMOTE-resampled training set. Final evaluation was conducted on the untouched test set, reporting metrics such as accuracy, confusion matrix, and classification report.



Overall, the project followed a comprehensive machine learning pipeline, including data preprocessing, exploratory analysis, model training, hyperparameter optimization, and rigorous evaluation. The final results demonstrated that a tuned Random Forest model combined with SMOTE provided the most robust and accurate solution for predicting customer churn.
