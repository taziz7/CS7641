import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv(filepath):

    names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
   # cancer = pd.read_csv(filepath, names=names)
    cancer = pd.read_csv(filepath, names=names)

    return cancer


def main():
    SHOW_PRINT = True
    url = "./data/breast-cancer-wisconsin.csv"
    print(">>>>123")
    cancer =  read_csv(url)
    if SHOW_PRINT:
        print(cancer.head())
# Preprocess the data
    cancer.replace('?',-99999, inplace=True)
#    print(cancer.axes)
    cancer.drop(['id'], 1, inplace=True)
    
# Plot histograms for each variable
    if SHOW_PRINT:
        cancer.hist(figsize = (10, 10))
        plt.show()
    # Create scatter plot matrix
        scatter_matrix(cancer, figsize = (18,18))
        plt.show()
# Create X and Y datasets for training
    X = np.array(cancer.drop(['class'], 1))
    y = np.array(cancer['class'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Testing Options
    seed = 8
    scoring = 'accuracy'

    # Define models to train
    models = []
    models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
    models.append(('SVM', SVC(gamma='auto')))

    # evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state = seed)

        cv_results =  model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

# Make predictions on validation dataset

    for name, model in models:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(name)
        print(accuracy_score(y_test, predictions))
        print(classification_report(y_test, predictions))       

    clf = SVC()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print(accuracy)

    example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
    example_measures = example_measures.reshape(len(example_measures), -1)
    prediction = clf.predict(example_measures)
    print(prediction)
    
    
if __name__ == "__main__":
    main()

	