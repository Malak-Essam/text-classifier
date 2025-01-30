from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the saved Naïve Bayes model and TF-IDF vectorizer
nb_tfidf = joblib.load("naive_bayes_tfidf.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize Flask app
app = Flask(__name__)

# Route to classify text
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the text from the incoming request
        input_data = request.json  # Assuming JSON input
        text = input_data['text']  # Extract the text field

        # Transform the text using the loaded TF-IDF vectorizer
        text_tfidf = vectorizer.transform([text])

        # Predict the class using the Naïve Bayes model
        prediction = nb_tfidf.predict(text_tfidf)

        # Return the predicted class as JSON response
        return jsonify({"prediction": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
