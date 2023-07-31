# Diabetes Prediction Model using Ensemble Learning
This repository contains a diabetes prediction model implemented using ensemble learning techniques. The model achieves an impressive accuracy of 99% and aims to predict whether an individual is likely to have diabetes based on various input attributes.

# Model Details
GitHub Repository: https://github.com/SaifSunny/Diabetes-Prediction-using-Ensemble
Live Demo: https://diabetes-prediction-using-ensemble.streamlit.app/
# How to Use the Application
The diabetes prediction application allows users to input their personal attributes and obtain a prediction result. To use the application, follow these steps:

1. Open the application using the live demo link mentioned above.
2. You will see a title and a brief description of the application.
3. Fill in the input attributes (age, gender, urea, creatinine ratio, HbA1c, cholesterol levels, triglycerides, HDL cholesterol, LDL cholesterol, VLDL cholesterol, and BMI) as requested in the form.
4. Choose one or multiple classifier models from the provided options. The models available for comparison are Random Forest, Naïve Bayes, Logistic Regression, K-Nearest Neighbors, Decision Tree, Gradient Boosting, LightGBM, XGBoost, Multilayer Perceptron, Artificial Neural Network, and Support Vector Machine.
5. Click the "Submit" button to view the results.
# Dataset Information
The diabetes prediction model is trained on the "Diabetes.csv" dataset. The dataset contains various attributes related to an individual's health and diabetes status. The class labels are transformed into numerical format (0 for "N" - No Diabetes, and 1 for "Y" - Yes Diabetes) for the model's training.

# Ensemble Learning
The diabetes prediction model leverages ensemble learning, a technique that combines multiple individual models to make more accurate predictions. The ensemble model is created using a VotingClassifier with the top 3 models, which are RandomForestClassifier, XGBClassifier, and LGBMClassifier.

# Model Performance
The ensemble model achieves an accuracy of 99% on the test set. Additionally, the application allows users to compare the performance of other selected models to see how they perform against the ensemble model.

# Conclusion
The diabetes prediction model using ensemble learning is a powerful tool to predict the likelihood of diabetes based on an individual's attributes. The model's high accuracy and the ability to compare it with other models make it a valuable tool in healthcare and research.

Feel free to explore the code and data in the GitHub repository and try out the live demo to experience the application firsthand.

For any inquiries or suggestions, please feel free to contact the repository owner. Happy predicting!




