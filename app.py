import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment import SentimentIntensityAnalyzer as NLTKSentimentIntensityAnalyzer
from afinn import Afinn
import tensorflow_hub as hub
from keras.models import load_model

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load the sentiment analysis model
model = load_model('./checkpoints/checkpoint')

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter the text for sentiment analysis:")

# Create a selectbox for choosing the sentiment analysis library or the custom model
library_info = {
    "TextBlob": "TextBlob is a simple library for processing textual data. It provides a sentiment analysis feature based on text polarity. TextBlob is easy to use and great for basic sentiment analysis tasks. The polarity in the TextBlob library is measured using a sentiment analysis approach that calculates the sentiment of a text as a numerical value within the range of -1 to 1. The polarity value indicates the sentiment of the text, with -1 representing a completely negative sentiment, 1 representing a completely positive sentiment, and 0 representing a neutral sentiment.",
    "VADER": "VADER (Valence Aware Dictionary and sEntiment Reasoner) is a sentiment analysis tool that uses a lexicon and rule-based approach to analyze text sentiment. It's widely used for social media sentiment analysis.",
    "NLTK": "NLTK (Natural Language Toolkit) offers the SentimentIntensityAnalyzer, which is a part of NLTK's suite for natural language processing tasks. NLTK is a comprehensive NLP library.",
    "AFINN": "AFINN is a wordlist-based sentiment analysis tool that assigns sentiment scores to words and calculates an overall score for text. It's simple and effective for sentiment analysis.",
    "Custom Model": "Choose a custom sentiment analysis model",
}

selected_option = st.selectbox("Select Analysis Method", list(library_info.keys()))

if selected_option == "Custom Model":
    if st.button("Analyze Sentiment"):
        if user_input:
            # Vectorize the user's input using the Universal Sentence Encoder
            user_input_vector = embed([user_input]).numpy()

            # Predict sentiment using your model
            prediction = model.predict(user_input_vector)

            # Define a threshold for classifying sentiment (you can adjust this)
            threshold = 0.5
            sentiment = "Positive" if prediction >= threshold else "Negative"

            st.write(f"Sentiment: {sentiment} (Probability: {prediction[0][0]:.2f})")
else:
    # Create a selectbox for choosing the sentiment analysis library
    selected_library = selected_option

    # Display library information on hover
    if selected_library in library_info:
        st.info(library_info[selected_library])

    # Define a function to analyze sentiment using various libraries
    def analyze_sentiment(text, library):
        if library == "TextBlob":
            analysis = TextBlob(text)
            polarity = analysis.sentiment.polarity
            return {'Polarity': polarity}
        elif library == "VADER":
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(text)
            return sentiment
        elif library == "NLTK":
            nltk_analyzer = NLTKSentimentIntensityAnalyzer()
            sentiment = nltk_analyzer.polarity_scores(text)
            return sentiment
        elif library == "AFINN":
            afinn = Afinn()
            score = afinn.score(text)
            return {'AFINN Score': score}

    # Analyze sentiment based on user choice
    if st.button("Analyze Sentiment"):
        if user_input:
            sentiment = analyze_sentiment(user_input, selected_library)

            if selected_library == "VADER":
                score = sentiment['compound']
                tmp = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
                st.write(f"Sentiment: {tmp}")
            elif selected_library == "TextBlob":
                polarity = sentiment['Polarity']
                st.write(f"Polarity: {polarity}")
            elif selected_library == "NLTK":
                score = sentiment['compound']
                tmp = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
                st.write(f"Sentiment: {tmp}")
            elif selected_library == "AFINN":
                afinn_score = sentiment['AFINN Score']
                st.write(f"AFINN Score: {afinn_score}")

            st.subheader("Sentiment Scores")
            st.bar_chart(sentiment)

            st.subheader("Raw Sentiment Scores")
            st.write(sentiment)

if 'sentiment' in st.session_state:
    st.write(f"Sentiment: {st.session_state.sentiment}")
