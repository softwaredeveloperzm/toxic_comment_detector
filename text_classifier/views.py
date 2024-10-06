from django.shortcuts import render
from django.http import HttpResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import time

# Load stopwords
stopwords_english = stopwords.words('english')

# Preprocessing function
def preprocess(corpus):
    for text in corpus:
        text = text.lower()
        text = re.sub(r'https?://[^\s\n\r]+', '', text)  # Remove URLs
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
        text = re.sub('\w*\d\w*', '', text)  # Remove words with numbers
        yield ' '.join([word for word in text.split() if word not in stopwords_english])

# Function to get model and log likelihood
def get_model(bow, target):
    log = np.log(probNB(bow, target, 1) / probNB(bow, target, 0))
    m = bow.dot(log)
    model = LogisticRegression(C=0.5, max_iter=500, solver='lbfgs', random_state=42).fit(m, target)
    return model, log

# Prediction function
def predict_new_text(text, vectorizer, bow, target):
    preprocessed_text = list(preprocess([text]))
    bow_input = vectorizer.transform(preprocessed_text)
    df_classification = pd.DataFrame()

    for i, j in enumerate(target.columns):
        model, log = get_model(bow, target[j])
        prediction = model.predict(bow_input.dot(log))
        df_classification = pd.concat([df_classification, pd.DataFrame({j: [prediction[0]]})], axis=1)
    
    return df_classification

# Load data and train the models
train_data = pd.read_csv('train.csv')
clean_comments = list(preprocess(train_data['comment_text']))
vectorizer = TfidfVectorizer(min_df=3, max_df=0.9)
bow = vectorizer.fit_transform(clean_comments)
target = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]




def get_input_text(request):
   
    if request.method == 'POST':
    
        text = request.POST.get('text', '')
        if text:
            predictions = predict_new_text(text, vectorizer, bow, target)
            if predict_new_text:
                return render(request, 'text_classifier/predict.html', {'predictions': predictions.to_dict(), 'text': text})
            else:
                return render(request, 'text_classifier/predict.html', {'predictions': predictions.to_dict(), 'text': text})

  

 

    
    return render(request, 'text_classifier/predict.html', {'predictions': None})




def probNB(bow, target, cat):
    p = np.array(bow[target == cat].sum(axis=0))
    return np.transpose((p + 1) / (p.sum() + bow.shape[1]))
