"""
Disaster Response Project: Web App

This web app shows results of the training dataset and also classifies new
input messages according to the trained model from the ML pipeline.

"""

# import libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar

import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text: str) -> list:
    """Clean messages text.

    Args:
        text (str): Text from messages.

    Returns:
        clean_tokens (list): Clean text tokens.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# Load data from SQLite (from ETL pipeline)
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("RawData", con=engine, index_col="id")

# Load model (from ML pipeline)
model = joblib.load("../models/classifier.pkl")


# Index webpage displays graphs and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # Extract data needed for graphs
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    categories = df.iloc[:, 3:]

    category_counts = []
    category_counts = categories.sum(axis=0).sort_values(ascending=True)

    # Create graphs
    graphs = [
        {
            "data": [Bar(x=genre_counts, y=genre_names, orientation="h")],
            "layout": {
                "title": "Distribution of message by genres",
                "yaxis": {"title": "Genre"},
                "xaxis": {"title": "Count"},
                "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
            },
        },
        {
            "data": [Bar(x=category_counts, y=category_counts.index, orientation="h")],
            "layout": {
                "title": "Classification of messages by category",
                "yaxis": {"title": "Category"},
                "xaxis": {"title": "Count"},
                "margin": {"l": 200, "r": 20, "t": 70, "b": 70},
                "height": 1000,
            },
        },
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# Web page that handles user query and displays model results
@app.route("/go")
def go():
    # Save user input in query
    query = request.args.get("query", "")

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template("go.html", query=query, classification_result=classification_results)


def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
