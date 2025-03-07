import streamlit as st
import json
import base64
from datetime import datetime
import uuid
import qrcode
import io
import hashlib

def generate_project_id(email):
    """Generate a unique project ID based on the user's email"""
    # Hash the email using SHA256 and take the first 8 characters
    hashed_email = hashlib.sha256(email.encode()).hexdigest()[:8].upper()
    return f"EMP-BURN-{hashed_email}"


def create_project_details_txt():
    """Create a TXT file content with project details"""
    email = "Your email"  # Your email
    project_id = generate_project_id(email)  # Generate project ID based on email

    # Get current date in a readable format
    current_date = datetime.now().strftime("%B %d, %Y")

    project_details = f"""App: Employee Burnout Analysis & Prediction Dashboard
Author: Pardha Saradhi Alapati
Project_ID: EMP-BURN-{project_id}
Version: 1.0.0
Created_Date: {current_date}
Features: [
  "Exploratory Data Analysis",
  "Burnout Prediction",
  "Model Evaluation",
  "Issue Reporting"
]
Technologies: [
  "Python",
  "Streamlit",
  "Scikit-learn",
  "Pandas",
  "Machine Learning"
]
Copyright_Notice: {{
  Year: 2025,
  Holder: Pardha Saradhi Alapati,
  Statement: This software is the intellectual property of Pardha Saradhi Alapati.
             All rights reserved © 2025 Pardha Saradhi Alapati.
             This work is protected by copyright laws and international treaties. Unauthorized reproduction, distribution, or modification of this software, in whole or in part, is strictly prohibited and may result in legal action.
}}
"""
    return project_details


def add_project_footer():
    """Add a footer with project details download option in TXT format"""
    project_details_txt = create_project_details_txt()  # Get project details as a string

    footer_html = """
    <div style='position: fixed; bottom: 0; left: 0; right: 0; background-color: #f8f9fa; border-top: 1px solid #e9ecef; padding: 0.5rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto; gap: 1rem;'>
            <span style='color: #495057; font-size: 0.9rem; font-weight: 500;'>
                All rights reserved © 2025 Pardha Saradhi Alapati
            </span>
        </div>
    </div>
    <div style='margin-bottom: 3rem;'></div>
    """

    # Display footer
    st.markdown(footer_html, unsafe_allow_html=True)

    # Add a download button for project details
    st.download_button(
        label="📄 Download Dashboard Details",
        data=project_details_txt,
        file_name="Dashboard_Details.txt",
        mime="text/plain",
    )