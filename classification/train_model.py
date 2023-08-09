import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import logging

logger = logging.getLogger(__name__)
class PreprocessingOptimization:
    def __init__(self, df):
        logger.info('Initialized PreprocessingOptimization')
        self.df = df
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self):
        logger.info('load and preprocessing data has started')
        if 'filename' in self.df.columns:
            self.df = self.df.drop(['filename'], axis=1)

        # remove periods from large numbers in all columns
        self.df = self.df.replace(r'\.', '', regex=True)

        # Remove rows with NaN values
        self.df = self.df.dropna()


        X = self.df.drop(['TARGET_WER', 'TARGET_CER', 'TARGET_LEVENSHTEIN-DISTANCE'], axis=1)
        y = self.df['TARGET_WER']


        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        self.le = LabelEncoder()
        y = self.le.fit_transform(y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        logger.info('train has started!')

        # Create and cross validate different models
        models = [SVC(), RandomForestClassifier(), MLPClassifier(max_iter=10000)]
        model_names = ['SVC', 'Random Forest', 'MLP Neural Network']

        self.predicted_labels = {}
        self.true_labels = {}  # Store the true labels for each model

        for model, name in zip(models, model_names):
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            cross_val = cross_val_score(model, self.X_train, self.y_train, cv=kfold)
            model.fit(self.X_train, self.y_train)

            # Transform the predicted labels to original label names
            original_predicted_labels = self.le.inverse_transform(model.predict(self.X_test))
            self.predicted_labels[name] = original_predicted_labels

            # Store the true labels
            original_true_labels = self.le.inverse_transform(self.y_test)
            self.true_labels[name] = original_true_labels

            print(f"{name} Train Score: {model.score(self.X_train, self.y_train)}")
            print(f"{name} Test Score: {model.score(self.X_test, self.y_test)}")
            print(f"{name} K-Fold Cross Validation Score: {np.mean(cross_val)}\n")

        with open("resources/reports/predicted_labels.txt", "w") as file:
            for model_name in self.predicted_labels.keys():
                file.write(f"Model: {model_name}\n")
                file.write("Predicted labels:\n")
                for label in self.predicted_labels[model_name]:
                    file.write(str(label) + "\n")
                file.write("True labels:\n")
                for label in self.true_labels[model_name]:
                    file.write(str(label) + "\n")

                file.write("\n")

