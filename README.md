# Disaster Response Pipeline Project

The project classifies disaster messages into several categories to allow quciker response in disaster situations.

### ETL pipeline

An ETL pipeline loads data from two CSV files and merges them together. The categories are then created and data is cleaned. The cleaned is saved to a SQLite database.

### ML Pipeline

A ML pipeline loads cleaned data from the SQLite database and prepares the text messages for the ML tasks. The multioutput classifier uses XGBoost to train the model and the evaluation results are displayed for each category. Model is saved for consumption in the web app.

### Web app

The web app is a simple Flash application which displays some details on the genres and categories of the training dataset. It includes a text box for user input of custom message which is then classified according to the categories.

### Instructions:

Libraries and virtual environment details are located in the `Pipfile` which can be used with `pipenv`.

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database

     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

   - To run ML pipeline that trains classifier and saves

     `python models/train_classifier.py data/DisasterResponse.db RawData models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.

   `python run.py`

3. Go to http://0.0.0.0:3001/

Project based on Udacity Data Science Nanodegree.
