import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Load data from CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge data into single dataframe
    df = pd.merge(messages, categories, on=["id"])

    # Extract and split the categories
    categories = pd.Series(df["categories"])
    categories = categories.str.split(";", expand=True)

    categories_series = pd.Series(df["categories"][0])
    categories_series = categories_series.str.split(";", expand=True)

    category_colnames = categories_series.apply(lambda x: x.str[:-2], axis=0)
    categories.columns = category_colnames.iloc[0]

    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = categories[column].astype(int)

    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    categories = df.iloc[:, 4:].columns

    for category in categories:
        mean_calc = df[category].mean()
        if mean_calc == 0 or mean_calc == 1:
            df.drop([category], axis=1, inplace=True)

    categories = df.iloc[:, 4:].columns

    for category in categories:
        max_calc = df[category].max()
        if max_calc > 1:
            indexNames = df[df[category] > 1].index
            df.drop(indexNames, inplace=True)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql("RawData", engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
        df = load_data(messages_filepath, categories_filepath)
        print(f"Dataframe shape: {df.shape}")

        print("Cleaning data...")
        df = clean_data(df)
        print(f"Dataframe shape: {df.shape}")

        print(f"Saving data...\n    DATABASE: {database_filepath}")
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()
