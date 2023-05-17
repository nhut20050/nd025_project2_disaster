# Disaster Response Pipeline 
by Huynh Nhut NGUYEN


## 1. Introductions
Data analysis and catastrophe categorization are also possible uses for the project. In order to create a simulated disaster response pipeline, the project employs data cleaning, preprocessing, model training, and classification assessment. Tokenizing and written descriptions of the situations will be used to train a model. We can observe how the model classifies a situation and analyzes the data by running the web demo.


## 2. Files Structure
```
📦Project 2
 ┣ 📂app
 ┃ ┣ 📂templates
 ┃ ┃ ┣ 📜go.html
 ┃ ┃ ┗ 📜master.html
 ┃ ┗ 📜run.py
 ┣ 📂data
 ┃ ┣ 📜disaster_categories.csv
 ┃ ┣ 📜disaster_messages.csv
 ┃ ┗ 📜process_data.py
 ┣ 📂models
 ┃ ┗ 📜train_classifier.py
 ┗ 📜README.md
 ```


## 3. Details
1. Pipeline ETL Use processing_data.py builds a pipeline for cleaning data that:
- To load raw datasets
- To join them together
- To clean the joined data
- To save the cleaned data to sqlite file for further use

2. Pipeline ML Use train_classifier.py to create a pipeline for machine learning that:
- To load data from saved sqlite database for model training
- To split the data to training and testing sets
- To set text processing and ML pipeline
- To train and tune a model by using GridSearchCV
- To validate the result
- To export the trained model to a local pickle file


## 4. How to run the pipeline?
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/nnguyen_disaster_response_data.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/nnguyen_disaster_response_data.db models/nnguyen_classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Copy the URL which is provided in the console to open the homepage
