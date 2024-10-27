from flask import Flask, jsonify, request, render_template, send_file  # Add send_file here

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
from keybert import KeyBERT
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import torch
import io
import csv

# Initialize Flask app
app = Flask(__name__)

# Load dataset
data = pd.read_csv('mental_health_dataset.csv')

# Load models and pipelines with GPU support if available
device = "cuda" if torch.cuda.is_available() else "cpu"
device_id = 0 if device == "cuda" else -1

sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"

# Mental health specific model for category classification
category_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Emotion Classification Pipeline
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
emotion_classifier = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer, device=device_id)

# Category Classification Pipeline
category_tokenizer = AutoTokenizer.from_pretrained(category_model_name)
category_model = AutoModelForSequenceClassification.from_pretrained(category_model_name)

# Define the categories and their descriptions for better classification
categories = {
    'Health Anxiety': 'worry about physical health, symptoms, illnesses',
    'Eating Disorder': 'concerns about food, weight, body image, eating patterns',
    'Anxiety': 'general worry, fear, panic, stress about various situations',
    'Depression': 'sadness, hopelessness, lack of interest, low mood',
    'Insomnia': 'difficulty sleeping, sleep patterns, tiredness',
    'Stress': 'feeling overwhelmed, pressure, tension',
    'Positive Outlook': 'hope, improvement, recovery, wellness',
    'Career Confusion': 'work-related stress, career decisions, job concerns'
}

# Create the classification pipeline with the categories
category_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device_id
)

# TF-IDF Vectorizer and Logistic Regression for Concern Classification
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['User Input'])

# Train the concern classifier
concern_classifier = LogisticRegression()
concern_classifier.fit(X, data['Extracted Concern'])

# Linear Regression Model for Intensity Scoring
intensity_regressor = LinearRegression()
intensity_regressor.fit(X, data['Intensity'])

# Keyword Extraction Model
# OldMin = 4
# OldMax = 6
# NewMin = 1
# NewMax = 10
kw_model = KeyBERT()
# def scale_value(OldValue):
#     OldRange = (OldMax - OldMin)  
#     NewRange = (NewMax - NewMin)  
#     NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
#     return NewValue
# Home route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/plot.png')
def plot_png():
    data = pd.read_csv("data.txt", header=None, names=["User Input", "Polarity", "Extracted Concern", "Category", "Category Confidence", "Intensity"])
    data['Time'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')  # Example: daily records

    plt.figure(figsize=(10, 6))
    plt.plot(data['Time'], data['Polarity'], label="Polarity", color="blue", marker="o")
    plt.plot(data['Time'], data['Intensity'], label="Intensity", color="orange", marker="o")
    plt.plot(data['Time'], data['Category Confidence'], label="Category Confidence", color="green", marker="o")

    # Configure plot details
    plt.title("Timeline of Sentiment Analysis")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.xticks(rotation=45)

    # Save plot to a BytesIO object
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()  # Close the plot to free memory
    return send_file(img, mimetype='image/png')
# Endpoint to process all NLP tasks in one request
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')

    # Emotion Classification
    emotion_result = emotion_classifier(text)[0]
    
    # Mapping labels to human-readable values
    if emotion_result['label'] == "LABEL_0":
        polarity = "Negative"
    elif emotion_result['label'] == "LABEL_1":
        polarity = "Neutral"
    elif emotion_result['label'] == "LABEL_2":
        polarity = "Positive"
    else:
        polarity = "Unknown"

    # Keyword Extraction
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words=None)
    extracted_keywords = [{"Keyword": keyword, "Score": score} for keyword, score in keywords]

    # Concern Classification
    input_vec = vectorizer.transform([text])
    concern = concern_classifier.predict(input_vec)[0]

    # Intensity Scoring
    predicted_intensity = intensity_regressor.predict(input_vec)[0]

    # Category Classification using zero-shot learning
    category_result = category_classifier(
        text,
        list(categories.keys()),
        hypothesis_template="This text is about {}."
    )
    
    # Get the category with highest confidence
    category = category_result['labels'][0]
    category_confidence = category_result['scores'][0]

    # Consolidate Results
    results = {
        "User Input": text,
        "Polarity": polarity,
        "Extracted Concern": extracted_keywords,
        "Category": category,
        "Category Confidence": f"{category_confidence:.2%}",
        # "Intensity": scale_value(round(predicted_intensity, 1))
        "Intensity": round(predicted_intensity, 1)
    }
    # Open the file in write mode
    with open("data.txt", "a") as file:
        file.write(f"{text},{polarity},{extracted_keywords[0]},{category},{category_confidence:.2%},{round(predicted_intensity, 1)}\n")

    return render_template("results.html", results=results)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
