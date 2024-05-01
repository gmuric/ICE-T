import pandas as pd
import json
import re
import pickle
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


def main():
    
    value_unknown = 0.5
    trial_tasks = ['ABDOMINAL', 'ADVANCED-CAD', 'ALCOHOL-ABUSE', 'ASP-FOR-MI', 'CREATININE', 'DIETSUPP-2MOS', 'DRUG-ABUSE', 'ENGLISH', 'HBA1C', 'MAJOR-DIABETES', 'MAKES-DECISIONS', 'MI-6MOS', "catalonia_independence_spanish", "climate_detection", "echtr_a", "medical_meadow_health_advice", "unfair_tos"]
    models = ["gpt-3.5-turbo-0125","gpt-4-0125-preview"]
    
    try:
        run_and_test_all(trial_tasks, models, value_unknown, classifiers)
    except:
        print("Error")

def run_and_test_all(trial_tasks, models, value_unknown, classifiers):
    for task in trial_tasks:
        for model in models:
            run_and_test(task, model,value_unknown,classifiers)

            
def run_and_test(task, model,value_unknown,classifiers):
    with open('train_test_data/' + task + '_' + model + '.pickle', 'rb') as handle:
        results = pickle.load(handle)

    train = pd.DataFrame([results['train_target']]).T.rename(columns={0:'target'})
    test = pd.DataFrame([results['test_target']]).T.rename(columns={0:'target'})

    train_answers_original = pd.DataFrame.from_dict(results['train_answers']).T
    test_answers_original = pd.DataFrame.from_dict(results['test_answers']).T
    train_answers = train_answers_original.copy()
    test_answers = test_answers_original.copy()

    for c in train_answers.columns:
        train_answers[c] = verbalize(train_answers[c],value_unknown)
    for c in test_answers.columns:
        test_answers[c] = verbalize(test_answers[c],value_unknown)

    train_answers.index = train_answers.index.map(int)
    test_answers.index = test_answers.index.map(int)
    train = train_answers.join(train['target'])
    test = test_answers.join(test['target'])

    ### Zero-shot
    zs = {}
    y_test = list(test['target'].values)
    y_pred = []
    if '0' in test.columns:
        tc = '0'
    if 0 in test.columns:
        tc = 0
    for x in test[tc]:
        if x == 0: y_pred.append(False)
        if x == 1: y_pred.append(True)
        if x == 0.5: y_pred.append(True)

    pr_m,rc_m,f1_m,sup = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)
    print('Zero shot')
    print('F1:' + str(f1_m))

    y_train = np.array(train['target']).astype(int)
    y_test = np.array(test['target']).astype(int)

    x_train = np.array(train.drop(columns=['target']))
    x_test = np.array(test.drop(columns=['target']))

    for name, classifier, params in classifiers:
        grid_search = GridSearchCV(classifier, param_grid=params, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        best_classifier = grid_search.best_estimator_
        y_pred = best_classifier.predict(x_test)
        pr_m,rc_m,f1_m,sup = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)
        print(name)
        print('F1:' + str(f1_m))
        print('')
        
if __name__ == "__main__":
    rs = 3
    classifiers = [
        ('K-Nearest Neighbors', KNeighborsClassifier(), {'n_neighbors': range(1, 32)}),
        ('Support Vector Machine', SVC(random_state=rs), {'probability':[True],'kernel': ['linear', 'rbf'], 'C': list(np.arange(0.1, 4, 0.1))}),
        ('Decision Tree', DecisionTreeClassifier(random_state=rs), {'max_depth': range(1, 11), 'criterion': ['gini', 'entropy', 'log_loss']}),
        ('Random Forest', RandomForestClassifier(random_state=rs), {'n_estimators': range(2, 40), 'max_depth': range(1, 11)}),
        ('Logistic Regression', LogisticRegression(random_state=rs), {'C': np.arange(0.01, 0.2,0.01), 'solver': ['newton-cg', 'lbfgs', 'liblinear']}),
        ('Gaussian Naive Bayes', GaussianNB(), {}),
        ('Multinomial Naive Bayes', MultinomialNB(), {'alpha': np.arange(0.00001, 0.005, 0.00002)}),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=rs), {'n_estimators': range(45,100), 'learning_rate': np.arange(0.1,0.3,0.02), 'max_depth': range(1, 4)}),
        ('ExtraTrees', ExtraTreesClassifier(random_state=rs), {'n_estimators': range(1,13),'criterion': ['gini', 'entropy', 'log_loss'],'max_depth': range(1,20, 1), 'bootstrap':[False,True]}),
        ('AdaBoost', AdaBoostClassifier(random_state=rs), {'estimator': [DecisionTreeClassifier(max_depth=i) for i in range(1,6)], 'n_estimators': range(5, 80, 5),'learning_rate': np.arange(0.01, 1, 0.05)}),
    ]
    main()
