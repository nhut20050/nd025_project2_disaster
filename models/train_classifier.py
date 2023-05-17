import sys
import pandas as pd
import nltk
nltk.download(["punkt", "wordnet", "stopwords"])
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    This function will load data which is saved from the process_data step.
    Return data according to our purposes
    Parameters:
    @database_filepath: a string of saved file path
    Return: X dataframe, Y dataframe, a list of category name
    """
    var_sqlite_path = "sqlite:///{}".format(database_filepath)
    eg = create_engine(var_sqlite_path)
    df = pd.read_sql_table(
        "nnguyen_disaster_response_data",
        eg
    )
    X_df = df["message"]
    Y_df = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X_df, Y_df, category_names


def tokenize(text):
    """
    Normalize, Tokenize and Lemmatize words, prepare for vectorizer
    Parameters:
    @text: a string of token text
    Return: a lemmed
    """
    # To normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # To tokenize the text
    words = word_tokenize(text)

    # To remove the stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # To lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w, pos="n").strip() for w in words]
    lemmed = [lemmatizer.lemmatize(w, pos="v").strip() for w in lemmed]

    return lemmed


def build_model():
    """
    This function describes a model's pipeline: text to vectors, tfidf transformation, and modeling
    Find the ideal settings with GridSearchCV.
    Parameters: None
    Return: a pipeline
    """
    # Initialize pipeline variable
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("moc", MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        "clf__estimator__n_estimators" : [50, 100]
    }
    
    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        verbose=3
    )
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates predicted result for test data
    Parameters:
    @model:
    @X_test: a dataframe for independant features
    @Y_test: a dataframe for labels
    @category_names: human label name
    Return: It just prints results and accuracy results
    """
    Y_pred = model.predict(X_test)

    # Determine the accuracy for every one of them.
    for i in range(len(category_names)):
        print(
            "Category:",
            category_names[i],
            "\n",
            classification_report(
                Y_test.iloc[:, i].values,
                Y_pred[:, i]
            )
        )
        print(
            "Accuracy of %25s: %.2f" % (
                category_names[i],
                accuracy_score(
                    Y_test.iloc[:, i].values,
                    Y_pred[:, i]
                )
            )
        )


def save_model(model, model_filepath):
    """
    This function will save our trained model to a local file
    Parameters:
    @model: our trained model
    @model_filepath: a string path of place which we want save to
    Return: Nothing
    """
    pickle.dump(
        model,
        open(model_filepath, "wb")
    )


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print("Building model...")
        model = build_model()
        
        print("Training model...")
        model.fit(X_train, Y_train)
        
        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print("Please provide the filepath of the disaster messages database "\
              "as the first argument and the filepath of the pickle file to "\
              "save the model to as the second argument. \n\nExample: python "\
              "train_classifier.py ../data/DisasterResponse.db classifier.pkl")


if __name__ == "__main__":
    main()