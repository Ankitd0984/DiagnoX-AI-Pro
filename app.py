from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
import pickle
import pandas as pd
import numpy as np
import os
from threading import Lock

# --- Initialize Flask ---
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

# --- Thread Lock for File Safety ---
file_lock = Lock()

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

MODEL_PATH = os.path.join(MODELS_DIR, 'disease_predictor.pkl')
USERS_PATH = os.path.join(DATA_DIR, 'users.csv')
MEDICATIONS_PATH = os.path.join(DATA_DIR, 'medications.csv')
TRAINING_DATA_PATH = os.path.join(DATA_DIR, 'Training.csv')

# Create data folder if missing
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    print(f"✅ Created data directory: {DATA_DIR}")

# Globals that will be initialized lazily
model = None
medications_df = None
symptoms = []

# --- SAFE ONE TIME INITIALIZATION ---
def initialize():
    global model, medications_df, symptoms

    # Avoid reload duplication
    if model is not None and medications_df is not None and symptoms:
        return

    print("🔄 Initializing model and data...")

    # Load model
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Model load error: {e}")
        model = None

    # Load Medications
    try:
        medications_df = pd.read_csv(MEDICATIONS_PATH)
        print("✅ Medications data loaded")
    except Exception as e:
        print(f"❌ Medications error: {e}")
        medications_df = None

    # Load symptom list from training CSV
    try:
        train_df = pd.read_csv(TRAINING_DATA_PATH)
        if 'Unnamed: 133' in train_df.columns:
            train_df = train_df.drop('Unnamed: 133', axis=1)
        symptoms = train_df.drop('prognosis', axis=1).columns.tolist()
        print(f"✅ Symptoms loaded: {len(symptoms)}")
    except Exception as e:
        print(f"❌ Training.csv error: {e}")
        symptoms = []


@app.before_request
def before_request():
    initialize()

# --- ROUTES ---
@app.route('/')
def root():
    return redirect(url_for('login'))

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', "").strip()
        email = request.form.get('email', "").strip()
        password = request.form.get('password', "").strip()

        if not username or not email or not password:
            flash("❌ All fields are required.", "danger")
            return redirect(url_for('signup'))

        if not os.path.exists(USERS_PATH):
            with file_lock:
                pd.DataFrame(columns=['username', 'email', 'password']).to_csv(USERS_PATH, index=False)

        with file_lock:
            users_df = pd.read_csv(USERS_PATH)

            if (users_df['username'] == username).any() or (users_df['email'] == email).any():
                flash("❌ Username or email already exists.", "danger")
                return redirect(url_for('signup'))

            new_user = pd.DataFrame([[username, email, password]],
                                    columns=['username', 'email', 'password'])
            users_df = pd.concat([users_df, new_user], ignore_index=True)
            users_df.to_csv(USERS_PATH, index=False)

        flash("✅ Signup successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', "").strip()
        password = request.form.get('password', "").strip()

        if not os.path.exists(USERS_PATH):
            flash("❌ No users registered yet. Please signup first.", "danger")
            return redirect(url_for('signup'))

        with file_lock:
            users_df = pd.read_csv(USERS_PATH)

        user = users_df[(users_df['username'] == username) &
                        (users_df['password'] == password)]
        if not user.empty:
            session['user'] = username
            flash(f"✅ Welcome back, {username}!", "success")
            return redirect(url_for('home'))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("✅ You have been logged out.", "success")
    return redirect(url_for('login'))

@app.route('/home')
def home():
    if "user" not in session:
        flash("🔒 Please login to access this page.", "warning")
        return redirect(url_for('login'))
    return render_template('index.html', symptoms=symptoms, user=session['user'])

# === MODIFIED PREDICT ROUTE ===
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or medications_df is None or not symptoms:
        return jsonify({'error': 'Server is not configured properly.'}), 500

    selected_symptoms = request.form.getlist('symptoms')
    
    if not selected_symptoms:
        return jsonify({'error': 'Select at least one symptom.'}), 400

    unknown_symptoms = set(selected_symptoms) - set(symptoms)
    if unknown_symptoms:
        return jsonify({'error': 'One or more selected symptoms are not supported.'}), 400

    try:
        input_data = np.zeros(len(symptoms))
        for symptom in selected_symptoms:
            input_data[symptoms.index(symptom)] = 1

        prediction = model.predict([input_data])[0]

        suggestion_row = medications_df[
            medications_df['Disease'].str.lower() == prediction.lower()
        ]
        suggestion = suggestion_row['Suggestion'].iloc[0] if not suggestion_row.empty else \
            "No specific suggestion found. Please consult a doctor."

        response = {
            'status': 'prediction',
            'prediction': prediction,
            'suggestion': suggestion
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}") # Added better error logging
        return jsonify({'error': str(e)}), 500
# === END OF MODIFIED ROUTE ===


# --- Run ---
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)