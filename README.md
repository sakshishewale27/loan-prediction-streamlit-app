💰 Loan Prediction App

A Machine Learning–powered Loan Prediction Web App built using Python, Streamlit, and Scikit-learn.
This app predicts whether a loan will be approved or not based on user input such as income, loan amount, and credit history.

🚀 Live Demo: https://loan-prediction-app-app-x9ccykmbduyprdzbtemjnv.streamlit.app/  

🧠 Project Overview

The Loan Prediction App uses a trained machine learning model (Random Forest / SVM) to predict the likelihood of loan approval.
It provides an easy-to-use web interface where users can enter details like applicant income, loan amount, and credit history — and instantly get a prediction result.

⚙️ Tech Stack

Python 3.12

Streamlit – Web framework for UI

Scikit-learn – Machine Learning model

Pandas & NumPy – Data processing

Joblib – Model serialization

📂 Project Structure
loan-prediction-app/
│
├── app.py                 # Streamlit web app
├── loan_model.joblib       # Trained ML model
├── preprocessors.pkl       # Preprocessing (scaler, encoders, etc.)
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

🚀 Installation & Setup

Clone the repository:

git clone https://github.com/yourusername/loan-prediction-app.git
cd loan-prediction-app


Create a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Run the app locally:

streamlit run app.py

☁️ Deployment

This project is deployed using Streamlit Community Cloud:

Push your code to a public GitHub repository.

Go to Streamlit Cloud
.

Connect your GitHub account and select your repo.

Add the file paths for:

app.py

requirements.txt

Click Deploy — your app will go live in a few minutes.

🔧 Usage

Open the app in your browser.

Enter the required details like:

Applicant Income

Coapplicant Income

Loan Amount

Loan Term

Credit History

Property Area

Click Predict to see if the loan is Approved or Rejected.

