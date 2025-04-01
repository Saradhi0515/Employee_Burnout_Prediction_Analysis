
# 💼 Employee Burnout Prediction Analysis

This repository contains a project focused on predicting employee burnout using various machine learning techniques. The goal is to analyze and predict the likelihood of employee burnout based on different features, which can help companies implement preventive measures and improve employee well-being.

## 📋 Project Overview

Employee burnout is a critical issue that affects productivity, job satisfaction, and overall well-being. This project aims to predict burnout levels based on factors such as workload, work-life balance, and personal well-being. By applying machine learning algorithms, the model identifies patterns that lead to burnout, helping companies take proactive steps in preventing it.

## 📊 Dataset

The dataset used in this project includes employee data with several features, such as:
- Employee ID
- Date of Joining
- Gender
- Company Type
- WFH Setup Available
- Designation
- Resource Allocation
- Mental Fatigue Score
- Burn Rate
- Etc.

Example:
- [Kaggle - Employee_Burnout_Prediction](https://www.kaggle.com/datasets/vijaysubhashp/employee-burnout-prediction)

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- Pickle
- Smtplib
- Email

## 🚀 Getting Started

### ✅ Prerequisites

To run this project locally, make sure you have the following installed:
- Python 3.x
- pip (Python package manager)

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### ▶️ Running the Code

1. Clone the repository:

```bash
git clone https://github.com/Saradhi0515/Employee_Burnout_Prediction_Analysis.git
```

2. Navigate to the project folder:

```bash
cd Employee_Burnout_Prediction_Analysis
```

3. Run The Model File To Get model.pkl:

```bash
python Model.py
```

4. Run The Utils Python Programs :

```bash
python mail.py
python project_details_footer.py
```

5. Run The Dashboard.py Using The Streamlit:

```bash
streamlit run path/to/Dashboard.py
```

### 🧩 Using Utility Python Files

```bash
mail.py                    # Adding Your Mail Details Which Is Required For Issues Section in Dashboard
project_details_footer.py  # Adding The Footer "All rights reserved" & Dashboard Details File Download Section
```

### 🖥️ Operating Dashboard

#### 📁 Dashboard Structure

```bash
|-- Dashboard
    |-- Home
    |-- EDA
    |-- Burnout Prediction
    |-- Model Evaluation
    |-- Report Issue
```

#### 🔍 Dashboard Operation

##### 📊 EDA

- Upload Your Dataset In csv File Format

##### 🔮 Burnout Prediction

- Enter Satisfaction Level (Eg:  0.40)
- Enter Last Evaluation (Eg:  0.55)
- Enter Number of Projects (Eg: 8)
- Enter Average Monthly Hour (Eg: 210)

##### Expected Output:
```bash
High Burnout Risk
Low Risk: 0.30, High Risk: 0.70
```

##### 🧪 Model Evaluation

- Upload Your Dataset In csv File Format
- Select The Model (Classification or Regression)

##### 📱 My App Link

- [Employee-Burnout](https://employee-burnout.streamlit.app/)

##### 📩 Report Issue

- If You Encounter An Issue, You Can Submit It Directly To The Mentioned Email In The Dashboard.py

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For any issues or inquiries, feel free to reach out via the [Issues](https://github.com/Saradhi0515/Employee_Burnout_Prediction_Analysis/issues) section.
