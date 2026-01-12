#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[9]:


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# In[10]:


def prepare_data(data):
    X = data['text']
    y = data['label']
    return X, y
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)
    return X_train, X_test, y_train, y_test


# In[11]:


def create_pipeline():
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    classifier = MultinomialNB()
    return tfidf_vectorizer, classifier
def train_model(tfidf_vectorizer, classifier, X_train, y_train):
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    classifier.fit(X_train_tfidf, y_train)
def predict(tfidf_vectorizer, classifier, X_test):
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_tfidf)
    return y_pred


# In[12]:


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, classification_rep


# In[14]:


def main():
    file_path = r"C:\Users\lavan\Downloads\news_dataset.csv"
    data = load_data(file_path)
    X, y = prepare_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    tfidf_vectorizer, classifier = create_pipeline()
    train_model(tfidf_vectorizer, classifier, X_train, y_train)
    y_pred = predict(tfidf_vectorizer, classifier, X_test)
    accuracy, conf_matrix, classification_rep = evaluate_model(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", classification_rep)
if __name__ == "__main__":
    main()


# In[15]:


ex:16


# In[16]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[17]:


nltk.download('vader_lexicon')


# In[21]:


def load_whatsapp_chat(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        chat_data = file.read()
        return chat_data


# In[22]:


def preprocess_chat_data(chat_data):
    processed_data = ''.join(char for char in chat_data if char.isalnum() or char.isspace())
    return processed_data
chat_filepath = r"C:\Users\lavan\Downloads\whatsapp_chat.txt"
chat_data = load_whatsapp_chat(chat_filepath)
preprocessed_data = preprocess_chat_data(chat_data)


# In[23]:


def perform_sentiment_analysis(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['compound']
chat_sentiment = perform_sentiment_analysis(preprocessed_data)
if chat_sentiment >= 0.05:
    overall_sentiment = 'Positive'
elif chat_sentiment <= -0.05:
    overall_sentiment = 'Negative'
else:
    overall_sentiment = 'Neutral'
print(f'Overall Sentiment: {overall_sentiment} (Sentiment Score: {chat_sentiment:.2f})')


# In[24]:


ex:17


# In[26]:


pip install pytesseract


# In[27]:


from PIL import Image
import pytesseract


# In[31]:


pytesseract.pytesseract.tesseract_cmd = r"c:\users\lavan\appdata\local\programs\python\python310\lib\site-packages (from pytesseract) (10.0.1)"


# In[32]:


def ocr_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print("Error during OCR: ", str(e))
        return None


# In[34]:


if __name__ == "__main__":
    image_path = r"C:\Users\lavan\Downloads\ocr.jpg" 
    extracted_text = ocr_image(image_path)
    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
    else:
        print("No text was extracted.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




