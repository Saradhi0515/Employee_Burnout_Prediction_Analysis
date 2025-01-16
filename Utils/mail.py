from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib

def send_issue_report(name, email, description):
    sender_email = "Replace with your email"  # Replace with your email
    sender_password = "Replace with your email password"  # Replace with your email password
    recipient_email = "Replace with your email to receive issues"  # Replace with your email to receive issues

    subject = "New Issue Reported in Burnout Dashboard"
    body = f"""
    Name: {name}
    Email: {email}

    Issue Description:
    {description}
    """

    try:
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Establish connection and send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
