import sys

# import libraries
import pandas as pd
from sqlalchemy import create_engine
import joblib
import re

# natural langauge libs
import nltk

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger", "stopwords"])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

stopwords_langauges = []


def load_data(database_filepath, database_table):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table(database_table, con=engine, index_col="id")

    X = df["message"].values
    Y = df.iloc[:, 3:].values
    categories = df.iloc[:, 3:].columns

    return X, Y, categories


def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for word in words:
        lemmatized_word = lemmatizer.lemmatize(word)
        clean_tokens.append(lemmatized_word)

    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(XGBClassifier(early_stopping_rounds=5, eval_metric="mlogloss"))),
        ]
    )
    parameters = {
        "vect__ngram_range": ((1, 1), (1, 2)),
        "tfidf__use_idf": (True, False),
        "clf__estimator__gamma": [0.5, 1, 1.5],
    }

    cv = GridSearchCV(pipeline, parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)

    index = 0
    for category in category_names:
        print(f"Category: {category} (index = {index})")
        evaluation_report = classification_report(Y_test[:, index], Y_pred[:, index], output_dict=True)
        index += 1
        report = evaluation_report["weighted avg"]
        print(f"Precision: {report['precision']:0.4f}, \tRecall: {report['recall']:0.4f}, \tF1 Score: {report['f1-score']:0.4f}")


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 4:
        database_filepath, database_table, model_filepath = sys.argv[1:]
        print(f"Loading data...\n    DATABASE: {database_filepath}")
        X, Y, category_names = load_data(database_filepath, database_table)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print(f"Saving model...\n    MODEL: {model_filepath}")
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument, the table name in the database as the "
            "second argument and the filepath of the pickle file to save "
            "the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db RawData "
            "classifier.pkl"
        )


if __name__ == "__main__":
    main()
