from flask import Flask, render_template, request, jsonify
import joblib
import scipy.sparse

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Load the trained model
    model = joblib.load('svm_model.pkl')

    # Load the CountVectorizers for each feature
    gene_vectorizer = joblib.load('gene_vector.pkl')
    variation_vectorizer = joblib.load('variation_vector.pkl')
    text_vectorizer = joblib.load('text_vector.pkl')


    # Get input values from the form
    gene = request.json.get('input1')
    variation = request.json.get('input2')
    text = request.json.get('input3')

    
    # Preprocess the user input using the corresponding CountVectorizers
    gene_vector = gene_vectorizer.transform([gene])
    variation_vector = variation_vectorizer.transform([variation])
    text_vector = text_vectorizer.transform([text])

    # Concatenate the preprocessed vectors into a single input vector
    input_vector = scipy.sparse.hstack((gene_vector, variation_vector, text_vector))

    # Make predictions using the preprocessed input
    prediction = model.predict(input_vector)

    # Return the prediction result as JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
