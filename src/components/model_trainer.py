import os
import sys
from dataclasses import dataclass
import dill

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("split data into Test and Train.")
            X_train, y_train,X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression" : LinearRegression(),
                "Support Vector Machines" : SVR(),
                "Decision Tree" : DecisionTreeRegressor(),
                "K-Nearest Neighbor" : KNeighborsRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Adaboost" : AdaBoostRegressor(),
                "XGBoost" : XGBRegressor(),
                "Catboost" : CatBoostRegressor(),
                "Gradient Boost" : GradientBoostingRegressor()
            }

            model_report : dict = evaluate_model(X_train,y_train,X_test,y_test,models)

            # To get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dictionary
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if(best_model_score < 0.6):
                raise CustomException("No best model found.")
            
            logging.info(f"Best found model on both training and testing dataset.")

            save_object(file_path= self.model_trainer_config.trained_model_file_path
                        , obj = best_model)
            
            predicted = best_model.predict(X_test)

            r2 = r2_score(y_test,predicted)

            return r2

        except Exception as ex:
            raise CustomException(ex,sys)
        
