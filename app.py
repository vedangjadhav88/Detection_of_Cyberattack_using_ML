import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import smtplib
from email.message import EmailMessage
import os
from matplotlib.backends.backend_pdf import PdfPages
import io

# Load pre-trained models
rf_model = joblib.load('random_forest_model.pkl')
lr_model = joblib.load('logistic_regression_model.pkl')
nn_model = joblib.load('neural_network_model.pkl')

# Label encoder for the 'Label' column
label_encoder = LabelEncoder()
label_encoder.fit(['BENIGN', 'DDoS'])

# Streamlit app
st.set_page_config(layout='wide')

# Apply custom CSS styles
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title('DDoS Attack Prediction Dashboard')

# Background image or GIF
st.markdown(
    """
    <style>
    .reportview-container {
        background: url('background.gif') no-repeat center center fixed; 
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Model selection
st.write("Select a machine learning model for predictions.")
model_option = st.selectbox("Choose a model", ["Random Forest", "Logistic Regression", "Neural Network"])

# File uploader
st.write('Upload your CSV file for prediction.')
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()  # Strip whitespace from column names

    expected_columns = ['Destination Port', 'Flow Duration', 'Total Fwd Packets',
                        'Total Backward Packets', 'Total Length of Fwd Packets',
                        'Total Length of Bwd Packets', 'Fwd Packet Length Max',
                        'Fwd Packet Length Min', 'Fwd Packet Length Mean',
                        'Fwd Packet Length Std', 'Bwd Packet Length Max',
                        'Bwd Packet Length Min', 'Bwd Packet Length Mean',
                        'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
                        'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
                        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
                        'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
                        'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std',
                        'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
                        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
                        'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s',
                        'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
                        'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
                        'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
                        'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
                        'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio',
                        'Average Packet Size', 'Avg Fwd Segment Size',
                        'Avg Bwd Segment Size', 'Fwd Header Length.1',
                        'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk',
                        'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
                        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
                        'Subflow Fwd Packets', 'Subflow Fwd Bytes',
                        'Subflow Bwd Packets', 'Subflow Bwd Bytes',
                        'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
                        'act_data_pkt_fwd', 'min_seg_size_forward',
                        'Active Mean', 'Active Std', 'Active Max', 'Active Min',
                        'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']

    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        st.error(f'Missing columns in the uploaded file: {", ".join(missing_columns)}')
    else:
        # Fill or drop NaN values
        data = data.dropna()  # Adjust this as necessary for your use case
        
        # Define features based on the expected columns
        X = data[expected_columns]
        
        # Encode categorical columns if necessary
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.factorize(X[col])[0]

        # Select model based on user choice
        if model_option == "Random Forest":
            model = rf_model
        elif model_option == "Logistic Regression":
            model = lr_model
        elif model_option == "Neural Network":
            model = nn_model

        # Make predictions based on the selected model
        predictions = model.predict(X)

        predicted_labels = label_encoder.inverse_transform(predictions)
        prediction_counts = pd.Series(predicted_labels).value_counts()

        # Save all plots to a single PDF
        pdf_filename = "prediction_report.pdf"
        with PdfPages(pdf_filename) as pdf:
            # Plot donut chart and scatter plot side by side
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader('Predictions Overview')
                if model_option == "Logistic Regression":
                    # For Logistic Regression, create a line chart instead of donut chart
                    benign_count = (predicted_labels == 'BENIGN').sum()
                    ddos_count = (predicted_labels == 'DDoS').sum()
                    
                    fig, ax = plt.subplots()
                    ax.plot(['BENIGN', 'DDoS'], [benign_count, ddos_count], marker='o', color='skyblue', label='Occurrences')
                    ax.fill_between(['BENIGN', 'DDoS'], [benign_count, ddos_count], color='skyblue', alpha=0.3)
                    ax.set_xlabel('Label')
                    ax.set_ylabel('Occurrences')
                    ax.set_title('Line Chart of BENIGN vs DDoS')
                    ax.set_facecolor("#0E1117")
                    ax.figure.set_facecolor("#0E1117")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    st.pyplot(fig)
                    fig.savefig("line_chart.png")
                    pdf.savefig(fig)
                else:
                    # Display donut chart only for Random Forest and Neural Network
                    st.subheader('DDoS vs BENING')
                    fig, ax = plt.subplots()
                    fig.patch.set_facecolor('#0E1117')
                    ax.set_facecolor('#0E1117')

                    wedges, texts, autotexts = ax.pie(
                        prediction_counts, 
                        labels=prediction_counts.index, 
                        autopct='%1.1f%%', 
                        startangle=90,
                        wedgeprops=dict(width=0.3, edgecolor='#0E1117'),
                        colors=plt.cm.Paired(range(len(prediction_counts)))
                    )
                    ax.axis('equal')
                    plt.setp(autotexts, size=10, weight="bold", color="white")
                    plt.setp(texts, color="white")
                    st.pyplot(fig)
                    fig.savefig("donut_chart.png")
                    pdf.savefig(fig)

            with c2:
                st.subheader('Scatter Plot')    
                fig = px.scatter(data, x='Flow Duration', y='Total Fwd Packets', color='Label', title='')
                st.plotly_chart(fig)
                
                # Save Plotly figure as a static image
                image_buffer = io.BytesIO(fig.to_image(format="png"))
                plt.figure(figsize=(10, 6))
                plt.imshow(plt.imread(image_buffer), aspect='auto')
                plt.axis('off')
                pdf.savefig()
                plt.close()

            # Plot ROC Curve and Flow Duration vs BENIGN/DDoS side by side
            c3, c4 = st.columns(2)
            
            with c3:
                st.subheader('ROC Curve')
                st.write("Generating ROC Curve...")
                y_true = data['Label'].dropna()  # Remove NaNs from y_true
                if set(y_true.unique()).issubset(set(label_encoder.classes_)):
                    y_true_encoded = label_encoder.transform(y_true)
                    fig, ax = plt.subplots()
                    y_scores = model.predict_proba(X)[:, 1]
                    fpr, tpr, _ = roc_curve(y_true_encoded, y_scores)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('ROC Curve')
                    ax.legend(loc="lower right")
                    ax.set_facecolor("#0E1117")
                    ax.figure.set_facecolor("#0E1117")
                    ax.xaxis.label.set_color("white")
                    ax.yaxis.label.set_color("white")
                    ax.title.set_color("white")
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    st.pyplot(fig)
                    fig.savefig("roc_curve.png")
                    pdf.savefig(fig)
            
            with c4:
                st.subheader('Histogram of Flow Duration')
                fig, ax = plt.subplots()
                benign_flows = data[data['Label'] == 'BENIGN']['Flow Duration']
                ddos_flows = data[data['Label'] == 'DDoS']['Flow Duration']
                ax.hist(benign_flows, bins=30, alpha=0.5, label='BENIGN')
                ax.hist(ddos_flows, bins=30, alpha=0.5, label='DDoS')
                ax.set_xlabel('Flow Duration')
                ax.set_ylabel('Frequency')
                ax.set_title('Flow Duration by Label')
                ax.legend(loc='upper right')
                ax.set_facecolor("#0E1117")
                ax.figure.set_facecolor("#0E1117")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                ax.title.set_color("white")
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                st.pyplot(fig)
                fig.savefig("flow_duration_histogram.png")
                pdf.savefig(fig)

        # Provide the download link for the PDF report
        st.subheader("Download the Prediction Report")
        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="Download Report as PDF",
                data=pdf_file,
                file_name="prediction_report.pdf",
                mime="application/pdf",
            )

        # Email form
        st.subheader("Send Results via Email")
        email = st.text_input("Enter your email address")
        send_button = st.button("Send")
        
        if send_button and email:
            # Create the email
            msg = EmailMessage()
            msg['Subject'] = 'Your DDoS Attack Prediction Results'
            msg['From'] = 'vedj.jadhav88@gmail.com'
            msg['To'] = email
            msg.set_content('Please find attached the prediction results.')
        
            # Attach the images
            for file in ["line_chart.png", "donut_chart.png", "scatter_plot.png", "roc_curve.png", "flow_duration_histogram.png", pdf_filename]:
                if os.path.exists(file):
                    with open(file, 'rb') as f:
                        file_data = f.read()
                        file_name = os.path.basename(file)
                        msg.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
        
            # Send the email
            try:
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                    smtp.login('vedj.jadhav88@gmail.com', '000321@Vj')
                    smtp.send_message(msg)
                st.success(f"Email sent successfully to {email}")
            except Exception as e:
                st.error(f"Failed to send email: {str(e)}")
