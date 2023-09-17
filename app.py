import pandas as pd
from sklearn.preprocessing import LabelEncoder
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import time

df = pd.read_csv("spam (1).csv", encoding='latin-1')
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
df.rename(columns={'v1': 'Target', 'v2': 'text'}, inplace=True)

# Create an instance of the LabelEncoder
label_encoder = LabelEncoder()

# Perform label encoding on your categorical column
df['Target'] = label_encoder.fit_transform(df['Target'])

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


df['text'] = df['text'].apply(transform_text)
# Create an instance of the TfidfVectorizer
vectorizer = TfidfVectorizer()

# Compute the TF-IDF values
tfidf_matrix = vectorizer.fit_transform(df['text'])

X = tfidf_matrix.toarray()
y = df['Target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MultinomialNB()

# Train the classifier on your training data
clf.fit(X_train, y_train)

st.title("Spam message detector")
st.write("Example to use:")
st.write("congratulation winner august prize draw call 09066660100 prize code 2309")
input_text=st.text_area("Enter your text here")

if st.button("CHECK"):

    cleaned_text=transform_text(input_text)
    cleaned_text_vector = vectorizer.transform([cleaned_text]).toarray()
    result = clf.predict(cleaned_text_vector)
    with st.spinner('Wait for it...'):
        time.sleep(5)

        if result==1:
            st.warning("The message is a spam message")
        else:
            st.success("This message is not spam message")
