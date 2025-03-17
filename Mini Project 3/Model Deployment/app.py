import streamlit as st
import joblib
import numpy as np
import textstat
from textblob import TextBlob

# ✅ Load models ONCE at startup
@st.cache_resource  # This caches models so they don't reload on every interaction
def load_models():
    return {
        "tfidf_vectorizer": joblib.load("tfidf_vectorizer.pkl"),
        "metadata_scaler": joblib.load("metadata_scaler.pkl"),
        "word2vec_model": joblib.load("word2vec_model.pkl").wv,  # ✅ Ensure .wv is used
        "glove_embeddings": joblib.load("glove_embeddings.pkl"),
        "model": joblib.load("best_model.pkl"),
        "threshold": joblib.load("best_threshold.pkl"),
    }

models = load_models()  # Load everything once

# ✅ Define Word2Vec feature extraction function
def get_word2vec_vector(tokens, word_vectors):
    vectors = [word_vectors[word] for word in tokens if word in word_vectors]
    return np.mean(vectors, axis=0) if vectors else np.zeros(word_vectors.vector_size)  # ✅ Dynamically get vector size

# ✅ Function to extract GloVe features
def get_glove_vector(tokens, glove_dict):
    vectors = [glove_dict[word] for word in tokens if word in glove_dict]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

# ✅ Function to extract metadata features from input text
def extract_metadata(text):
    word_count = len(text.split())
    char_count = len(text)
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)
    sentiment = TextBlob(text).sentiment
    sentiment_polarity = sentiment.polarity
    sentiment_subjectivity = sentiment.subjectivity

    return np.array([word_count, char_count, flesch_reading_ease, smog_index, sentiment_polarity, sentiment_subjectivity]).reshape(1, -1)

# ✅ Function to preprocess input text into features
def preprocess_input(user_text):
    # Use preloaded models
    tfidf_vectorizer = models["tfidf_vectorizer"]
    metadata_scaler = models["metadata_scaler"]
    word2vec_model = models["word2vec_model"]
    glove_model = models["glove_embeddings"]

    # TF-IDF transformation
    tfidf_features = tfidf_vectorizer.transform([user_text]).toarray()  # ✅ Shape: (1, tfidf_vocab_size)

    # Word2Vec transformation
    word2vec_features = get_word2vec_vector(user_text.split(), word2vec_model).reshape(1, -1)  # ✅ Shape: (1, word2vec_vector_size)

    # GloVe transformation
    glove_features = get_glove_vector(user_text.split(), glove_model).reshape(1, -1)

    # Metadata transformation
    metadata_features = extract_metadata(user_text)  # ✅ Shape: (1, 6)

    # Check shape before concatenation
    print(f"Metadata Features: {metadata_features.shape}")  # Should be (1, 6)
    print(f"TF-IDF Features: {tfidf_features.shape}")  # Should match training dim
    print(f"Word2Vec Features: {word2vec_features.shape}")  # Should be (1, 300) if vector_size=300

    # Concatenate all features
    final_features = np.hstack((metadata_features, tfidf_features, word2vec_features, glove_features))

    # Apply MinMaxScaler to the full feature set
    final_features_scaled = models["metadata_scaler"].transform(final_features)

    return final_features_scaled

st.title("🔍 Suicidal Ideation Detection")
st.markdown("Enter a text message below to check if it indicates suicidal ideation.")

# Text input
user_input = st.text_area("Enter text here:", "")

if st.button("Analyze"):
    if user_input:
        # ✅ Use preloaded models
        model = models["model"]
        threshold = models["threshold"]

        # Preprocess input
        features = preprocess_input(user_input)

        # Check shape before prediction
        print(f"Final Features Shape: {features.shape}")  # Should match training input shape

        # Predict
        prob = model.predict_proba(features)[:, 1][0]
        prediction = int(prob >= threshold)

        # Show results
        st.subheader("Results:")
        st.write(f"**Prediction:** {'🛑 Suicidal message alert!!!' if prediction == 1 else '✅ This message is safe.'}")
        st.write(f"**Confidence Score:** {round(float(prob), 4)}")

    else:
        st.warning("⚠️ Please enter some text to analyze.")


