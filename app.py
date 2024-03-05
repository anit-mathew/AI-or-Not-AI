# Import necessary libraries
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('ai_detection_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = [request.form['text']]
        input_tfidf = tfidf_vectorizer.transform(input_text)
        prediction = model.predict(input_tfidf)
        result = "AI Generated" if prediction[0] == 1 else "Human Generated"
        return render_template('index.html', result=result, input_text=input_text[0])

if __name__ == '__main__':
    app.run(debug=True)
