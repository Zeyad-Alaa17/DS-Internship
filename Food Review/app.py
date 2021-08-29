import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

model_name = 'rf_model.pk'
vectorizer_name = 'tfidf_vectorizer.pk'
loaded_model = pickle.load(open(model_name, 'rb'))
loaded_vect = pickle.load(open(vectorizer_name, 'rb'))


def clean_review(review):
    stop_words = set(stopwords.words('english'))
    stop_words.remove('not')
    stemmer = SnowballStemmer("english")
    review = review.lower()
    review = re.sub(r'<.*br.*>', '', review)
    review = re.sub(r'[^\w\s]', '', review)
    stemmed = []
    for word in review.split():
        if word in stop_words:
            continue
        stemmed.append(stemmer.stem(word))
        stemmed.append(" ")
    return ["".join(stemmed).strip()]


def raw_test(review, model, vectorizer):
    review_c = clean_review(review)
    embedding = vectorizer.transform(review_c)
    prediction = model.predict(embedding)
    return "Positive" if prediction == 1 else "Negative"


def run():
    st.title("Amazon food review")
    text = st.text_area('Please enter review')
    if st.button("Predict"):
        output = raw_test(text, loaded_model, loaded_vect)
        st.success(f"This is a {output} review")


if __name__ == "__main__":
    run()
