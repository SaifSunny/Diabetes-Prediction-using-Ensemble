import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.title('Diabetes Prediction Application')
st.write('''
         Please fill in the attributes below, then hit the Predict button
         to get your results. 
         ''')

st.header('Input Attributes')
age = st.slider('Your Age (Years)', min_value=0.0, max_value=100.0, value=50.0, step=1.0)
st.write(''' ''')
gen = st.radio("Your Gender", ('Male', 'Female'))
st.write(''' ''')
# gender conversion
if gen == "Male":
    gender = 1
else:
    gender = 0

urea = st.slider('Urea', min_value=0.0, max_value=100.0, value=50.0, step=0.1)
st.write(''' ''')
cr = st.slider('Creatinine Ratio(Cr)', min_value=0.0, max_value=1000.0, value=500.0, step=1.0)
st.write(''' ''')
hb = st.slider('HbA1c', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
chol = st.slider('Cholesterol (Chol)', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
tg = st.slider('Triglycerides(TG) Cholesterol', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
hdl = st.slider('HDL Cholesterol', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
ldl = st.slider('LDL Cholesterol', min_value=0.0, max_value=20.0, value=10.0, step=0.1)
st.write(''' ''')
vldl = st.slider('VLDL Cholesterol', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
st.write(''' ''')
bmi = st.slider('BMI', min_value=0.0, max_value=50.0, value=25.0, step=0.1)
st.write(''' ''')

selected_models = st.multiselect("Choose Classifier Models", ('Random Forest', 'Naïve Bayes', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree', 'Gradient Boosting', 'LightGBM', 'XGBoost', 'Multilayer Perceptron', 'Artificial Neural Network', 'Support Vector Machine'))
st.write(''' ''')

# Initialize an empty list to store the selected models
models_to_run = []

# Check which models were selected and add them to the models_to_run list
if 'Random Forest' in selected_models:
    models_to_run.append(RandomForestClassifier())

if 'Naïve Bayes' in selected_models:
    models_to_run.append(GaussianNB())

if 'Logistic Regression' in selected_models:
    models_to_run.append(LogisticRegression())

if 'K-Nearest Neighbors' in selected_models:
    models_to_run.append(KNeighborsClassifier())

if 'Decision Tree' in selected_models:
    models_to_run.append(DecisionTreeClassifier())

if 'Gradient Boosting' in selected_models:
    models_to_run.append(GradientBoostingClassifier())

if 'Support Vector Machine' in selected_models:
    models_to_run.append(SVC())

if 'LightGBM' in selected_models:
    models_to_run.append(LGBMClassifier())

if 'XGBoost' in selected_models:
    models_to_run.append(XGBClassifier())

if 'Multilayer Perceptron' in selected_models:
    models_to_run.append(MLPClassifier())

if 'Artificial Neural Network' in selected_models:
    models_to_run.append(MLPClassifier(hidden_layer_sizes=(100,), max_iter=100))



user_input = np.array([age, gender, urea, cr, hb, chol, tg, hdl, vldl,
                       ldl, bmi]).reshape(1, -1)

# import dataset
def get_dataset():
    data = pd.read_csv('Diabetes.csv')
    # Transforming class into numerical format
    data['CLASS'] = data['CLASS'].apply(lambda x: 0 if x == 'N' else 1)

    # Transforming 	Gender into numerical format
    data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'M' else 0)

    # Calculate the correlation matrix
    # corr_matrix = data.corr()

    # Create a heatmap of the correlation matrix
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    # plt.title('Correlation Matrix')
    # plt.xticks(rotation=45)
    # plt.yticks(rotation=0)
    # plt.tight_layout()

    # Display the heatmap in Streamlit
    # st.pyplot()

    return data

def generate_model_labels(model_names):
    model_labels = []
    for name in model_names:
        words = name.split()
        if len(words) > 1:
            # Multiple words, use initials
            label = "".join(word[0] for word in words)
        else:
            # Single word, take the first 3 letters
            label = name[:3]
        model_labels.append(label)
    return model_labels

if st.button('Submit'):
    df = get_dataset()

    # fix column names
    df.columns = (["id", "pation_no", "gender", "age", "urea", "cr",
                   "hb", "chol", "tg", "hdl", "ldl",
                   "vldl", "bmi", "target"])

    # Split the dataset into train and test
    X = df.drop(['target','id','pation_no'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create two columns to divide the screen
    left_column, right_column = st.columns(2)


    # Left column content
    with left_column:
        # Create a VotingClassifier with the top 3 models
        ensemble = VotingClassifier(
            estimators=[('rf', RandomForestClassifier()), ('xgb', XGBClassifier()), ('gb', LGBMClassifier())],
            voting='hard')

        # Fit the voting classifier to the training data
        ensemble.fit(X_train, y_train)

        # Make predictions on the test set
        model_predictions = ensemble.predict(user_input)

        # Evaluate the model's performance on the test set
        ensamble_accuracy = accuracy_score(y_test, ensemble.predict(X_test))
        ensamble_precision = precision_score(y_test, ensemble.predict(X_test))
        ensamble_recall = recall_score(y_test, ensemble.predict(X_test))
        ensamble_f1score = f1_score(y_test, ensemble.predict(X_test))

        if model_predictions == 1:
            st.write(f'According to Ensemble Model You have a **Very High Chance (1)** of Heart Disease.')
        else:
            st.write(f'According to Ensemble Model You have a **Very Low Chance (0)** of Heart Disease.')

        st.write('Ensemble Model Accuracy:', ensamble_accuracy)
        st.write('Ensemble Model Precision:', ensamble_precision)
        st.write('Ensemble Model Recall:', ensamble_recall)
        st.write('Ensemble Model F1 Score:', ensamble_f1score)
        st.write('------------------------------------------------------------------------------------------------------')


    # Right column content
    with right_column:

        for model in models_to_run:
            # Train the selected model
            model.fit(X_train, y_train)

            # Make predictions on the test set
            model_predictions = model.predict(user_input)

            # Evaluate the model's performance on the test set
            model_accuracy = accuracy_score(y_test, model.predict(X_test))
            model_precision = precision_score(y_test, model.predict(X_test))
            model_recall = recall_score(y_test, model.predict(X_test))
            model_f1score = f1_score(y_test, model.predict(X_test))

            if model_predictions == 1:
                st.write(f'According to {type(model).__name__} Model You have a **Very High Chance (1)** of Heart Disease.')
            else:
                st.write(f'According to {type(model).__name__} Model You have a **Very Low Chance (0)** of Heart Disease.')

            st.write(f'{type(model).__name__} Accuracy:', model_accuracy)
            st.write(f'{type(model).__name__} Precision:', model_precision)
            st.write(f'{type(model).__name__} Recall:', model_recall)
            st.write(f'{type(model).__name__} F1 Score:', model_f1score)
            st.write('------------------------------------------------------------------------------------------------------')

    # Initialize lists to store model names and their respective performance metrics
    model_names = ['Ensemble']
    accuracies = [ensamble_accuracy]
    precisions = [ensamble_precision]
    recalls = [ensamble_recall]
    f1_scores = [ensamble_f1score]

    # Loop through the selected models to compute their performance metrics
    for model in models_to_run:
        model_names.append(type(model).__name__)
        model.fit(X_train, y_train)
        model_predictions = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, model_predictions))
        precisions.append(precision_score(y_test, model_predictions))
        recalls.append(recall_score(y_test, model_predictions))
        f1_scores.append(f1_score(y_test, model_predictions))

    # Create a DataFrame to store the performance metrics
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1 Score': f1_scores
    })

    # Get the model labels
    model_labels = generate_model_labels(metrics_df['Model'])

    # Plot the comparison graphs
    plt.figure(figsize=(12, 10))

    # Accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(model_labels, metrics_df['Accuracy'], color='skyblue')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1)

    # Precision comparison
    plt.subplot(2, 2, 2)
    plt.bar(model_labels, metrics_df['Precision'], color='orange')
    plt.title('Precision Comparison')
    plt.ylim(0, 1)

    # Recall comparison
    plt.subplot(2, 2, 3)
    plt.bar(model_labels, metrics_df['Recall'], color='green')
    plt.title('Recall Comparison')
    plt.ylim(0, 1)

    # F1 Score comparison
    plt.subplot(2, 2, 4)
    plt.bar(model_labels, metrics_df['F1 Score'], color='purple')
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1)

    # Adjust layout to prevent overlapping of titles
    plt.tight_layout()

    # Display the graphs in Streamlit
    st.pyplot()