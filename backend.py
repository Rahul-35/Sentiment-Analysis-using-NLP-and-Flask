import nltk
import numpy as np
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words=set(stopwords.words('english'))

data=pd.read_csv('C:\\Users\\Dell\\Desktop\\review.csv')

data['combined']=data["review_headline"] +" "+ data["review_body"]

df=data[['combined','sentiment']]

df=df.rename(columns={'combined':'review'})

lema=WordNetLemmatizer()


def clean_text(text):
    # Remove tokens like [[VIDEOID:...]]
    text = re.sub(r"\[\[.*?\]\]", "", text)

    # Remove HTML tags like <br/>, <div>, etc.
    text = re.sub(r"<.*?>", "", text)

    # Remove HTML entities like &#34;, &amp;, etc.
    text = re.sub(r"&[^;\s]+;", "", text)

    # Remove leftover brackets, slashes, or weird punctuation
    text = re.sub(r"[\[\]\\/]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def pre_process_data(orig):
    if isinstance(orig, str):  
        clean_data = clean_text(orig.lower())  
    else:
        clean_data = clean_text(str(orig).lower())  
    tokens=word_tokenize(clean_data)
    tokens=[lema.lemmatize(word,pos='v') for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

df['tokenized']=df['review'].apply(pre_process_data)

tfidf=TfidfVectorizer(max_features=5000,ngram_range=(1,2))

X=tfidf.fit_transform(df["tokenized"].apply(lambda tokens: " ".join(tokens)))
y=df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = LogisticRegression(max_iter=1000, solver='liblinear', penalty='l1')
model.fit(X_train, y_train)

y_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)

def user_input(user_text):
    user_token=pre_process_data(user_text)
    tokenized_joined = [" ".join(user_token)]
    user_vector=tfidf.transform(tokenized_joined)
    proba=model.predict_proba(user_vector)[0]
    return proba[1]


accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

conf=confusion_matrix(y_test,y_pred)
print(f"Confusion matrix:{conf}\n")

lloss=log_loss(y_test,y_proba)
print(f"Log Loss : {lloss}")