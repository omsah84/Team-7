import streamlit as st
import joblib
import pandas as pd
from scipy.sparse import hstack

# Load models and vectorizers
svc_model = joblib.load('../model/svc_model.pkl')
tfidf_word = joblib.load('../model/tfidf_word_vectorizer.pkl')
tfidf_char = joblib.load('../model/tfidf_char_vectorizer.pkl')

# App title
st.title("📝 Product Review Sentiment Analyzer")
st.subheader("Analyze individual or bulk product reviews for sentiment")

# Tabs for Single or Batch Prediction
tab1, tab2 = st.tabs(["🔹 Single Review", "📄 Bulk Reviews (CSV)"])

# ----------- TAB 1: SINGLE REVIEW -----------
with tab1:
    st.markdown("### ✏️ Enter a review manually or use an example:")
    
    sample_reviews = {
        "Excellent product, totally worth it!": "Positive",
        "Terrible experience, not recommended at all.": "Negative",
        "It works okay, but could be better.": "Neutral-ish",
        "Love the quality and delivery was super fast.": "Positive",
        "Very disappointing, broke after two days.": "Negative"
    }

    example_review = st.selectbox("Choose a sample review (optional):", [""] + list(sample_reviews.keys()))

    # Auto-fill input if example is chosen
    user_input = st.text_area("Your Review:", value=example_review if example_review else "", height=150)

    if st.button("🔍 Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a review.")
        else:
            word_feat = tfidf_word.transform([user_input])
            char_feat = tfidf_char.transform([user_input])
            final_feat = hstack([char_feat, word_feat])
            prediction = svc_model.predict(final_feat)[0]

            if prediction == 'Positive':
                st.success("🎉 Sentiment: **Positive**")
            else:
                st.error("😞 Sentiment: **Negative**")

# ----------- TAB 2: BULK PREDICTION -----------
with tab2:
    st.markdown("### 📂 Upload a CSV file with a column named `review_body`")

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            if 'review_body' not in df.columns:
                st.error("❌ The uploaded CSV must have a 'review_body' column.")
            else:
                st.info(f"✅ {len(df)} reviews loaded. Processing...")

                word_feat = tfidf_word.transform(df['review_body'])
                char_feat = tfidf_char.transform(df['review_body'])
                final_feat = hstack([char_feat, word_feat])

                predictions = svc_model.predict(final_feat)
                df['Predicted Sentiment'] = predictions

                st.success("✅ Sentiment analysis complete!")
                st.dataframe(df[['review_body', 'Predicted Sentiment']])

                csv_download = df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Predictions", csv_download, "sentiment_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing file: {e}")
