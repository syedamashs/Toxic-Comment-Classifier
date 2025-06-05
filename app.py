from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load toxic comment classifier once
classifier = pipeline("text-classification", model="unitary/toxic-bert", return_all_scores=True)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    comment = ""

    if request.method == 'POST':
        comment = request.form['comment']
        predictions = classifier(comment)[0]

        # Sort and filter predictions
        predictions.sort(key=lambda x: x['score'], reverse=True)
        result = [(pred['label'], round(pred['score'], 3)) for pred in predictions if pred['score'] > 0.5]

        if not result:
            result = [("Not Toxic", 1.0)]

    return render_template('index.html', result=result, comment=comment)

if __name__ == '__main__':
    app.run(debug=True)
