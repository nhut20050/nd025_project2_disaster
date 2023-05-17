import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function uses to loading data from raw to pandas dataframe
    Then join them by categories
    Parameters:
    @messages_filepath: a string path of messages csv file
    @categories_filepath: a string path of categories csv file
    Return: joined dataframe by categories
    """
    # To read csv files by using pandas
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # To join those dataframes above by categories
    joined_df = messages_df.merge(
        categories_df,
        on=["id"],
        how="inner"
    )

    return joined_df

def clean_data(df):
    """
    This function will clean the joined data above
    Parameters:
    @df: a pandas dataframe
    Return: a dataframe after cleaned according to our purposes
    """
    # To make a dataframe with each of the 36 category columns.
    temp_df = df["categories"]
    temp_df = temp_df.str.split(";", expand=True)

    # To get the first row of the categories_df dataframe
    the_first_row = temp_df.iloc[0]

    # To rename the temp_df columns
    temp_cols = [col[:-2] for col in the_first_row]
    temp_df.columns = temp_cols

    for col in temp_df:
        # Each value should be set to the string"s final character
        temp_df[col] = [cell[-1:] for cell in temp_df[col]]

        # We should convert value of each cell to numeric value
        temp_df[col] = pd.to_numeric(temp_df[col])

        # We should replace 2s with 1s
        temp_df["related"] = temp_df["related"].replace(to_replace=2, value=1)

    # To add additional category columns in place of the categories column in the dataframe
    df = df.drop("categories", axis=1)

    # To add the new "categories" dataframe to the existing dataframe
    df = pd.concat([df, temp_df], axis=1)

    # To remove duplicate rows
    result_df = df.drop_duplicates(keep="first")

    return result_df

def save_data(df, database_filename):
    """
    This function will save our dataframe to a local file
    Parameters:
    @df: a dataframe
    @database_filename: a string of file path
    Return: Nothing
    """
    path = "sqlite:///"+ str(database_filename)
    eg = create_engine(path)
    df.to_sql(
        "nnguyen_disaster_response_data",
        eg,
        index=False,
        if_exists="replace"
    )
    pass

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print("Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}"
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)
        
        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)
        
        print("Cleaned data saved to database!")
    
    else:
        print("Please provide the filepaths of the messages and categories "\
              "datasets as the first and second argument respectively, as "\
              "well as the filepath of the database to save the cleaned data "\
              "to as the third argument. \n\nExample: python process_data.py "\
              "disaster_messages.csv disaster_categories.csv "\
              "DisasterResponse.db")


if __name__ == "__main__":
    main()