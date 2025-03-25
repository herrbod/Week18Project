# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    
    # -------- WRITE YOUR CODE HERE --------
    
    # Step 1: Define arguments for train data, test data, model output, and RandomForest hyperparameters. Specify their types and defaults.  
    parser.add_argument("--train_data", type=uri_folder, help="Path to train data")
    parser.add_argument("--test_data", type="uri_folder", help="Path to test data")
    parser.add_argument('--n_estimators', type=int, default=100,
                        help='The number of trees in the forest')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.')                    
    parser.add_argument("--model_output", type=mlflow_model, help="Path of output model")
    args = parser.parse_args()

    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''

    # -------- WRITE YOUR CODE HERE --------

    # Step 2: Read the train and test datasets from the provided paths using pandas. Replace '_______' with appropriate file paths and methods.
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")
    
    # Step 3: Split the data into features (X) and target (y) for both train and test datasets. Specify the target column name.
    # Dropping the label column and assigning it to y_train
    y_train = train_df["price"].values  # 'price' is the target variable in this case study

    # Dropping the 'price' column from train_df to get the features and converting to array for model training
    X_train = train_df.drop("price", axis=1).values

    # Dropping the label column and assigning it to y_test
    y_test = test_df["price"].values  # 'price' is the target variable for testing

    # Dropping the 'price' column from test_df to get the features and converting to array for model testing
    X_test = test_df.drop("price", axis=1).values

    # Step 4: Initialize the RandomForest Regressor with specified hyperparameters, and train the model using the training data.
    tree_model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    tree_model = tree_model.fit(X_train, y_train)

    # Step 5: Log model hyperparameters like 'n_estimators' and 'max_depth' for tracking purposes in MLflow.
    mlflow.log_param("model", "RandomForest")  # Provide the model name
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Step 6: Predict target values on the test dataset using the trained model, and calculate the mean squared error.
    yhat_test = tree_model.predict(X_test)
    
    # Step 7: Log the MSE metric in MLflow for model evaluation, and save the trained model to the specified output path.  
    mse = mean_squared_error(y_test, yhat_test)
    print('Mean Squared Error of RandomForest Regressor on test set: {:.2f}'.format(mse))
    mlflow.log_metric("MSE", float(mse))  # Log the MSE

    mlflow.sklearn.save_model(sk_model=tree_model, path=args.model_output)  # Save the model

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

