import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score
import pickle
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import sys
import os

# Add the parent directory to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from Utils.mail import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils.project_details_footer import *

def main():
    # Load pre-trained model (replace with your actual model path)
    MODEL_PATH = 'X:\\Employee_Burnout_Prediction_Analysis\\Model\\model.pkl'
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    
    st.title("Employee Burnout Analyzing & Prediction Dashboard")
    st.write("The Employee Burnout Analysis and Prediction Dashboard leverages data analysis and predictive modeling to help organizations monitor employee well-being, identify early signs of burnout, and take proactive steps to improve workplace conditions and prevent burnout.")

    # Create a container for the main content
    main_content = st.container()
    
    # Create a container for the footer that will always be at the bottom
    footer_container = st.container()

    # Sidebar navigation
    menu = ["Home", "EDA", "Burnout Prediction", "Model Evaluation"] # , "Report Issue"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.subheader("Welcome to the Burnout Analyzing & Prediction Dashboard")
        st.write("Dear user, please use the navigation menu to explore different sections of the dashboard for detailed insights and analysis.")

    elif choice == "EDA":
        st.subheader("Exploratory Data Analysis")
        uploaded_file = st.file_uploader("Upload Dataset", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview")
            st.dataframe(df.head())
            st.write("### Data Summary")
            st.write(df.describe())

            st.write("### Missing Values")
            st.write(df.isnull().sum())

            st.write("### Scatter Plots")
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            sns.scatterplot(x=df['Designation'], y=df['Burn Rate'], ax=axs[0])
            axs[0].set_title('Designation vs Burn Rate')
            axs[0].set_xlabel('Designation')
            axs[0].set_ylabel('Burn Rate')

            sns.scatterplot(x=df['Resource Allocation'], y=df['Burn Rate'], ax=axs[1])
            axs[1].set_title('Resource Allocation vs Burn Rate')
            axs[1].set_xlabel('Resource Allocation')

            sns.scatterplot(x=df['Mental Fatigue Score'], y=df['Burn Rate'], ax=axs[2])
            axs[2].set_title('Mental Fatigue Score vs Burn Rate')
            axs[2].set_xlabel('Mental Fatigue Score')

            st.pyplot(fig)

            # Additional EDA Visualizations
            st.write("### EDA Visualizations")
            plt.figure(figsize=(10,8))
            sns.countplot(x="Gender", data=df)
            plt.title("Plot Distribution of Gender")
            st.pyplot(plt)

            plt.figure(figsize=(10,8))
            sns.countplot(x="Company Type", data=df)
            plt.title("Plot Distribution of Company Type")
            st.pyplot(plt)

            plt.figure(figsize=(10,8))
            sns.countplot(x="WFH Setup Available", data=df)
            plt.title("Plot Distribution of WFH Setup Available")
            st.pyplot(plt)

            burn_st = df.loc[:,'Date of Joining':'Burn Rate']
            burn_st = burn_st.select_dtypes([int, float])
            for i, col in enumerate(burn_st.columns):
                fig = px.histogram(burn_st, x=col, title="Plot Distribution of "+col, color_discrete_sequence=['indianred'])
                st.plotly_chart(fig)

            fig = px.line(df, y="Burn Rate", color="Designation", title="Burn rate on the basis of Designation", color_discrete_sequence=px.colors.qualitative.Pastel1)
            st.plotly_chart(fig)

            fig = px.line(df, y="Burn Rate", color="Gender", title="Burn rate on the basis of Gender", color_discrete_sequence=px.colors.qualitative.Pastel1)
            st.plotly_chart(fig)

            fig = px.line(df, y="Mental Fatigue Score", color="Designation", title="Mental fatigue vs Designation", color_discrete_sequence=px.colors.qualitative.Pastel1)
            st.plotly_chart(fig)

            st.write("### Correlation Heatmap")
            for col in df.select_dtypes(include=['object', 'category']).columns:
                df[col] = LabelEncoder().fit_transform(df[col])
            # Compute the correlation matrix
            corr = df.corr()
            # Set figure size for seaborn
            sns.set(rc={'figure.figsize': (14, 12)})
            # Create and show the heatmap using Plotly
            heatmap = px.imshow(corr, text_auto=True, aspect='auto')
            st.plotly_chart(heatmap)
            pass

    elif choice == "Burnout Prediction":
        st.subheader("Predict Employee Burnout")
        st.write("Enter employee data below:") 
        # Input fields for the required features
        satisfaction_level = st.number_input("Satisfaction Level", 0.0, 1.0, step=0.01)
        last_evaluation = st.number_input("Last Evaluation", 0.0, 1.0, step=0.01)
        number_project = st.number_input("Number of Projects", 1, 10, step=1)
        average_monthly_hours = st.number_input("Average Monthly Hours", 0, 400, step=1)

        # Collect only the 4 features needed for prediction
        inputs = np.array([[satisfaction_level, last_evaluation, number_project, average_monthly_hours]])

        if st.button("Predict Burnout"):
            prediction = model.predict(inputs)
            prediction_proba = model.predict_proba(inputs)

            st.write("### Prediction")
            if prediction[0] == 1:
                st.error("High Burnout Risk")
            else:
                st.success("Low Burnout Risk")

            st.write("### Prediction Probability")
            st.write(f"Low Risk: {prediction_proba[0][0]:.2f}, High Risk: {prediction_proba[0][1]:.2f}")
            pass

    elif choice == "Model Evaluation":
        st.subheader("Model Evaluation Metrics")
        st.write("Evaluate the model using classification or regression approaches.")
        uploaded_file = st.file_uploader("Upload Dataset for Model Evaluation", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            Label_encode = preprocessing.LabelEncoder()
            df['GenderLabel'] = Label_encode.fit_transform(df['Gender'].values)
            df['Company_TypeLabel'] = Label_encode.fit_transform(df['Company Type'].values)
            df['WFH_Setup_Available'] = Label_encode.fit_transform(df['WFH Setup Available'].values)
            Columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'GenderLabel', 'Company_TypeLabel', 'WFH_Setup_Available']
            if not all(col in df.columns for col in Columns):
                st.error(f"One or more required columns are missing in the dataset: {Columns}")
            else:
                x = df[Columns]
                y = df['Burn Rate']
                # Impute missing values
                imputer = SimpleImputer(strategy='mean')
                x_imputed = imputer.fit_transform(x)
                y.fillna(y.mean(), inplace=True)

            # PCA for dimensionality reduction
                pca = PCA(0.95)
                x_pca = pca.fit_transform(x_imputed)

                x_train, x_test, y_train, y_test = train_test_split(x_pca, y, test_size=0.25, random_state=10)

            # Select approach: Classification or Regression
                mode = st.radio("Select Evaluation Mode", ["Classification", "Regression"])

                if mode == "Classification":
                    # Convert Burn Rate to categorical labels
                    bins = [0, 0.3, 0.7, 1]  # Define bin edges (low, medium, high)
                    labels = [0, 1, 2]        # Assign class labels
                    y_train_binned = pd.cut(y_train, bins=bins, labels=labels, include_lowest=True)
                    y_test_binned = pd.cut(y_test, bins=bins, labels=labels, include_lowest=True)

                    # Classification
                    model = RandomForestClassifier(random_state=42)
                    model.fit(x_train, y_train_binned)
                    y_pred = model.predict(x_test)

                    st.write("### Classification Metrics")
                    st.write("#### Confusion Matrix")
                    cm = confusion_matrix(y_test_binned, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    st.pyplot(plt)

                    st.write("#### Classification Report")
                    report = classification_report(y_test_binned, y_pred, output_dict=True)
                    st.write(pd.DataFrame(report).transpose())

                    st.write("#### Accuracy")
                    accuracy = accuracy_score(y_test_binned, y_pred)
                    st.write(f"Accuracy: {accuracy * 100:.2f}%")

                elif mode == "Regression":
                    # Regression
                    model = RandomForestRegressor(random_state=42)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)

                    st.write("### Regression Metrics")
                    st.write("#### R2 Score")
                    r2 = r2_score(y_test, y_pred)
                    st.write(f"R2 Score: {r2:.2f}")

                    st.write("#### Mean Absolute Error")
                    mae = np.mean(np.abs(y_test - y_pred))
                    st.write(f"Mean Absolute Error: {mae:.2f}")

                    st.write("#### Actual vs Predicted Burn Rate")
                    plt.figure(figsize=(10, 6))
                    plt.scatter(y_test, y_pred, alpha=0.7)
                    plt.xlabel("Actual Burn Rate")
                    plt.ylabel("Predicted Burn Rate")
                    plt.title("Actual vs Predicted Burn Rate")
                    st.pyplot(plt)
                    pass

    # elif choice == "Report Issue":
    #     st.subheader("Report an Issue")
    #     st.write("If you encounter any issues with the dashboard, please let us know below.")

    #     # Input fields for issue reporting
    #     name = st.text_input("Your Name")
    #     email = st.text_input("Your Email")
    #     description = st.text_area("Describe the Issue")

    #     if st.button("Submit Issue"):
    #         if name and email and description:
    #             sent = send_issue_report(name, email, description)
    #             if sent:
    #                 st.success("Thank you for reporting the issue. We will get back to you shortly.")
    #             else:
    #                 st.error("Failed to send the issue report. Please try again later.")
    #         else:
    #             st.error("Please fill out all the fields before submitting.")

    # Add the project footer at the bottom
    add_project_footer()

if __name__ == '__main__':
    main()
