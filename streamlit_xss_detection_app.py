import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from scipy.sparse import hstack
import re

# Function definitions
def extract_keyword_presence(payload, keywords):
    return [int(keyword in payload.lower()) for keyword in keywords]

def url_length(payload):
    return len(payload)

def preprocess_payload(payload):
    payload = re.sub(r'%3C', '<', payload, flags=re.IGNORECASE)
    payload = re.sub(r'%3E', '>', payload, flags=re.IGNORECASE)
    payload = re.sub(r'%27', "'", payload, flags=re.IGNORECASE)
    payload = re.sub(r'%22', '"', payload, flags=re.IGNORECASE)
    return payload.lower()

def prepare_features(data, keywords, vectorizer=None):
    data['url_length'] = data['payload'].apply(url_length)
    for keyword in keywords:
        data[f'keyword_{keyword}'] = data['payload'].apply(lambda x: int(keyword in x.lower()))
    
    cleaned_payloads = data['payload'].apply(preprocess_payload)
    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=500)
        ngram_features = vectorizer.fit_transform(cleaned_payloads)
    else:
        ngram_features = vectorizer.transform(cleaned_payloads)
    
    manual_features = data[['url_length'] + [f'keyword_{k}' for k in keywords]]
    X_combined = hstack([ngram_features, manual_features])
    return X_combined, vectorizer

def plot_label_distribution(data):
    fig, ax = plt.subplots()
    sns.countplot(data=data, x='label', palette='viridis', ax=ax)
    ax.set_title("Label Distribution")
    st.pyplot(fig)

def plot_keyword_presence(data, keywords):
    keyword_sums = data[[f'keyword_{k}' for k in keywords]].sum()
    fig, ax = plt.subplots()
    keyword_sums.plot(kind='bar', color='skyblue', edgecolor='black', ax=ax)
    ax.set_title("Keyword Presence in Payloads")
    st.pyplot(fig)

def plot_url_length_distribution(data):
    fig, ax = plt.subplots()
    sns.histplot(data['url_length'], bins=30, kde=True, color="orange", ax=ax)
    ax.set_title("URL Length Distribution")
    st.pyplot(fig)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

# Streamlit app layout
st.title("XSS Detection Model")
keywords = ['alert', 'script', 'http', 'contenteditable', 'onpaste', 'php']

# Upload data
st.sidebar.header("Upload Datasets")
train_file = st.sidebar.file_uploader("Upload Training Dataset", type=["csv"])
test_file = st.sidebar.file_uploader("Upload Testing Dataset", type=["csv"])

if train_file and test_file:
    # Load data
    st.subheader("Data Overview")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    st.write("### Training Data")
    st.write(train_data.head())
    st.write("### Testing Data")
    st.write(test_data.head())

    # Feature preparation
    st.subheader("Feature Engineering")
    X_combined, vectorizer = prepare_features(train_data, keywords)
    y = train_data['label']

    # Balance data
    st.write("Balancing Data...")
    training_data = pd.concat([pd.DataFrame(X_combined.toarray()), pd.DataFrame(y, columns=['label'])], axis=1)
    majority_class = training_data[training_data.label == 'xssdom']
    minority_class = training_data[training_data.label == 'xssreflected']
    nonattack_class = training_data[training_data.label == 'nonattack']
    
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)
    nonattack_upsampled = resample(nonattack_class, replace=True, n_samples=len(majority_class), random_state=42)
    balanced_data = pd.concat([majority_class, minority_upsampled, nonattack_upsampled])
    X_balanced = balanced_data.drop('label', axis=1)
    y_balanced = balanced_data['label']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    # Train model
    st.write("Training Model...")
    model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(X_test)
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.write("### Confusion Matrix")
    plot_confusion_matrix(y_test, y_pred, model.classes_)

    # Visualizations
    st.subheader("Data Visualizations")
    st.write("### Label Distribution")
    plot_label_distribution(train_data)

    st.write("### Keyword Presence")
    plot_keyword_presence(train_data, keywords)

    st.write("### URL Length Distribution")
    plot_url_length_distribution(train_data)

    # Predict on test data
    st.subheader("Testing Data Predictions")
    X_combined_testing, _ = prepare_features(test_data, keywords, vectorizer)
    y_testing_pred = model.predict(X_combined_testing)
    test_data['predicted_label'] = y_testing_pred
    st.write(test_data.head())

    # Save predictions
    st.write("Saving predictions...")
    output_path = 'logtesting_results.csv'
    test_data.to_csv(output_path, index=False)
    st.write(f"Predictions saved to: {output_path}")