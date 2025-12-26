# app.py
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import re
import random
import pandas as pd
import numpy as np
import csv
import warnings
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from difflib import get_close_matches
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------- Flask setup ----------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret")  # consider using env var in production
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# ---------- Load Data ----------
# Ensure these paths exist relative to where you run app.py
TRAINING_PATH = os.path.join("Data", "Training.csv")
TESTING_PATH = os.path.join("Data", "Testing.csv")
MASTER_DESC = os.path.join("MasterData", "symptom_Description.csv")
MASTER_SEV = os.path.join("MasterData", "symptom_severity.csv")
MASTER_PRECAUTION = os.path.join("MasterData", "symptom_precaution.csv")

for p in (TRAINING_PATH, TESTING_PATH, MASTER_DESC, MASTER_SEV, MASTER_PRECAUTION):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

training = pd.read_csv(TRAINING_PATH)
testing = pd.read_csv(TESTING_PATH)

# Remove trailing .1, .2 etc. from column names if present and remove duplicate columns
training.columns = training.columns.str.replace(r"\.\d+$", "", regex=True)
testing.columns = testing.columns.str.replace(r"\.\d+$", "", regex=True)
training = training.loc[:, ~training.columns.duplicated()]
testing = testing.loc[:, ~testing.columns.duplicated()]

# Feature columns (all except the target 'prognosis')
if 'prognosis' not in training.columns:
    raise KeyError("'prognosis' column missing from Training.csv")

cols = list(training.columns[:-1])  # feature names as a python list
x = training[cols]
y = training['prognosis']

# Label encode target
le = preprocessing.LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split and model training
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.33, random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(x_train, y_train)

# ---------- Dictionaries and helpers ----------
severityDictionary = {}
description_list = {}
precautionDictionary = {}

# Build a mapping from symptom name -> feature index used for the model input vector
# Use the same columns order as the DataFrame used to train the model
symptoms_dict = {symptom: idx for idx, symptom in enumerate(cols)}

def getDescription():
    with open(MASTER_DESC, newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) >= 2:
                description_list[row[0].strip()] = row[1].strip()

def getSeverityDict():
    with open(MASTER_SEV, newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if len(row) >= 2:
                try:
                    severityDictionary[row[0].strip()] = int(row[1].strip())
                except:
                    # skip rows with bad integer conversion
                    pass

def getprecautionDict():
    with open(MASTER_PRECAUTION, newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Expecting: disease,prec1,prec2,prec3,prec4  (or fewer)
            if len(row) >= 2:
                key = row[0].strip()
                values = [c.strip() for c in row[1:]]
                # pad/truncate to 4 if needed
                precautionDictionary[key] = values

getSeverityDict()
getDescription()
getprecautionDict()

# Synonyms mapping for free-text detection
symptom_synonyms = {
    "stomach ache":"stomach_pain","belly pain":"stomach_pain","tummy pain":"stomach_pain",
    "loose motion":"diarrhea","motions":"diarrhea","high temperature":"fever",
    "temperature":"fever","feaver":"fever","coughing":"cough","throat pain":"sore_throat",
    "cold":"chills","breathing issue":"breathlessness","shortness of breath":"breathlessness",
    "body ache":"muscle_pain"
}

def extract_symptoms(user_input, all_symptoms):
    """
    Extract symptoms from free-text user_input.
    all_symptoms: iterable of canonical symptom names (e.g. cols)
    Returns a deduplicated list of canonical symptom keys.
    """
    extracted = []
    text = user_input.lower().replace("-", " ")
    # match synonyms (phrases)
    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            extracted.append(mapped)
    # match multi-word symptom names by direct containment
    for symptom in all_symptoms:
        readable = symptom.replace("_", " ").lower()
        if readable in text:
            extracted.append(symptom)
    # fuzzy match individual words to symptom names
    words = re.findall(r"\w+", text)
    readable_symptoms = [s.replace("_"," ") for s in all_symptoms]
    for word in words:
        close = get_close_matches(word, readable_symptoms, n=1, cutoff=0.8)
        if close:
            # map back to canonical symptom name
            matched_readable = close[0]
            for sym in all_symptoms:
                if sym.replace("_"," ") == matched_readable:
                    extracted.append(sym)
    # return unique items preserving insertion order
    seen = set()
    unique = []
    for e in extracted:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique

def predict_disease(symptoms_list):
    """
    Build model input vector from symptoms_list and return:
      (disease_name (str), confidence_percent (float), probability_array (np.array))
    """
    input_vector = np.zeros(len(symptoms_dict), dtype=int)
    for symptom in symptoms_list:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    # predict_proba returns probabilities in the order of model.classes_
    pred_proba = model.predict_proba([input_vector])[0]       # array of probabilities
    best_index = int(np.argmax(pred_proba))                   # position index (0..n-1)
    # map index to the encoded class label and then decode to string label
    encoded_class = model.classes_[best_index]
    disease = le.inverse_transform([encoded_class])[0]
    confidence = round(float(pred_proba[best_index]) * 100.0, 2)
    return disease, confidence, pred_proba

quotes = [
    "🌸 Health is wealth, take care of yourself.",
    "💪 A healthy outside starts from the inside.",
    "☀️ Every day is a chance to get stronger and healthier.",
    "🌿 Take a deep breath, your health matters the most.",
    "🌺 Remember, self-care is not selfish."
]

# ------------------ State Machine (Flask routes) ------------------
@app.route('/')
def index():
    session.clear()
    session['step'] = 'welcome'
    # Make sure you have templates/index.html
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json.get('message', '').strip()
    step = session.get('step', 'welcome')

    # replicate each console step
    if step == 'welcome':
        session['step'] = 'name'
        return jsonify(reply="🤖 Welcome to HealthCare ChatBot!\n👉 What is your name?")
    elif step == 'name':
        session['name'] = user_msg
        session['step'] = 'age'
        return jsonify(reply="👉 Please enter your age:")
    elif step == 'age':
        session['age'] = user_msg
        session['step'] = 'gender'
        return jsonify(reply="👉 What is your gender? (M/F/Other):")
    elif step == 'gender':
        session['gender'] = user_msg
        session['step'] = 'symptoms'
        return jsonify(reply="👉 Describe your symptoms in a sentence:")
    elif step == 'symptoms':
        symptoms_list = extract_symptoms(user_msg, cols)
        if not symptoms_list:
            return jsonify(reply="❌ Could not detect valid symptoms. Please describe again:")
        session['symptoms'] = symptoms_list.copy()
        disease, conf, _ = predict_disease(symptoms_list)
        session['pred_disease'] = disease
        session['step'] = 'days'
        return jsonify(reply=f"✅ Detected symptoms: {', '.join(symptoms_list)}\n👉 For how many days have you had these symptoms?")
    elif step == 'days':
        session['days'] = user_msg
        session['step'] = 'severity'
        return jsonify(reply="👉 On a scale of 1–10, how severe is your condition?")
    elif step == 'severity':
        session['severity'] = user_msg
        session['step'] = 'preexist'
        return jsonify(reply="👉 Do you have any pre-existing conditions?")
    elif step == 'preexist':
        session['preexist'] = user_msg
        session['step'] = 'lifestyle'
        return jsonify(reply="👉 Do you smoke, drink alcohol, or have irregular sleep?")
    elif step == 'lifestyle':
        session['lifestyle'] = user_msg
        session['step'] = 'family'
        return jsonify(reply="👉 Any family history of similar illness?")
    elif step == 'family':
        session['family'] = user_msg
        # guided disease-specific questions
        disease = session.get('pred_disease')
        # find the row in training that corresponds to this disease
        disease_rows = training[training['prognosis'] == disease]
        if disease_rows.shape[0] == 0:
            # fallback if disease not in training
            session['disease_syms'] = []
        else:
            disease_symptoms = list(disease_rows.iloc[0][:-1].index[disease_rows.iloc[0][:-1] == 1])
            session['disease_syms'] = disease_symptoms
        session['ask_index'] = 0
        session['step'] = 'guided'
        return ask_next_symptom()
    elif step == 'guided':
        # record yes/no for the last asked symptom
        idx = session.get('ask_index', 0) - 1
        ds = session.get('disease_syms', [])
        if idx >= 0 and idx < len(ds):
            if user_msg.strip().lower() in ('yes','y'):
                # avoid duplicates
                if ds[idx] not in session['symptoms']:
                    session['symptoms'].append(ds[idx])
        return ask_next_symptom()
    elif step == 'final':
        # already answered all guided questions
        return final_prediction()
    else:
        # unknown step -> reset conversation
        session.clear()
        session['step'] = 'welcome'
        return jsonify(reply="Session reset. 🤖 Hello — what's your name?")

def ask_next_symptom():
    i = session.get('ask_index', 0)
    ds = session.get('disease_syms', [])
    if i < min(8, len(ds)):
        sym = ds[i]
        session['ask_index'] = i + 1
        return jsonify(reply=f"👉 Do you also have {sym.replace('_',' ')}? (yes/no):")
    else:
        session['step'] = 'final'
        return final_prediction()

def final_prediction():
    symptoms = session.get('symptoms', [])
    disease, conf, _ = predict_disease(symptoms)
    about = description_list.get(disease, 'No description available.')
    precautions = precautionDictionary.get(disease, [])
    text = (f"                        Result                            \n"
            f"\n🩺 Based on your answers, you may have **{disease}**\n"
            f"\n🔎 Confidence: {conf}%\n📖 About: {about}\n")
    if precautions:
        text += "\n\n🛡️ Suggested precautions:\n" + "\n\n".join(f"{i+1}. {p}" for i,p in enumerate(precautions))
    text += "\n\n\n💡 " + random.choice(quotes)
    name = session.get('name', 'User')
    text += f"\n\n\nThank you for using the chatbot. Wishing you good health, {name}!"
    return jsonify(reply=text)

# Run server
if __name__ == '__main__':
    # app.run(debug=True)  # use debug=True during development
    app.run(host='0.0.0.0', port=5000)
