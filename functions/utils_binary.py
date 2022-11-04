import pandas as pd
from rdkit.Chem import PandasTools
import numpy as np
from rdkit import DataStructs
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from json import JSONEncoder
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from time import time
from sklearn.model_selection import KFold, StratifiedKFold
from skopt import BayesSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from joblib import dump, load
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score
import math
import sys,os

def check_extention(file, encode = 'latin-1'):
    """ check the file extention and convert to ROMol if necessary """

    if file[-3:] == "sdf":       
        imported_file = PandasTools.LoadSDF(file, smilesName='SMILES', includeFingerprints=False)
        return imported_file
    
    elif file[-4:] == "xlsx":
        imported_file = pd.read_excel(file)
        return imported_file
        
    elif file[-3:] == "csv":
        imported_file = pd.read_csv(file, encoding=encode)
        return imported_file
    
    else:
        return ("file extension not supported, supported extentions are: csv, xlsx and sdf")

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Reporting util for different optimizers
def report_perf(optimizer, X, y, title="model", callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f " + u"\u00B1"+" %.3f") % (time() - start, len(optimizer.cv_results_['params']), best_score, best_score_std))    
    print('Best parameters:')
    print(best_params)
    print()
    return best_params

def metrics_binary(clf, x_ext, y_ext):
    """ Calculates metrics using confusion matrix """

    y_pred = clf.predict(x_ext)

    cm = confusion_matrix(y_ext, y_pred, labels=clf.classes_)

    metrics = {'ACC':[], 'Sen':[], 'Spe':[], 'Precision':[], 'BACC':[], 'MCC':[], 'F1':[], 'Classification Error':[], 'AUC':[]}


    tp = cm[1, 1]
    tn = cm[0, 0]
    fn = cm[1, 0]
    fp = cm[0, 1]

    Accuracy = (tp + tn) / (tp+fn+fp+tn)
    Sensitivity = tp / (tp + fn) #recall
    Specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    bacc = (Sensitivity + Specificity) / 2
    mcc = (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    f1 = 2 * (precision * Sensitivity) / (precision + Sensitivity)
    classificationError = (fp + fn) / (tp + tn)
    auc = roc_auc_score(y_ext, y_pred)


    metrics['ACC'].append(Accuracy)
    metrics['Sen'].append(Sensitivity)
    metrics['Spe'].append(Specificity)
    metrics['Precision'].append(precision)
    metrics['BACC'].append(bacc)
    metrics['MCC'].append(mcc)
    metrics['F1'].append(f1)
    metrics['Classification Error'].append(classificationError)
    metrics['AUC'].append(auc)


    metrics = pd.DataFrame(metrics)
    metrics = metrics.T
    metrics.columns = ['value' for col_name in metrics.columns]

    return metrics

def defineXY(testSize = 0.20, trainSize = 0.80, randomState = 4):

    # load y
    with open('./data/curated_data/y/binary/y_binary.json', 'r') as y_file:
        y_data = json.load(y_file)
        y = np.asarray(y_data)
    
    # loop through generated fp
    for file in os.listdir('./data/fp/binary/'):

        if file.endswith('json'):
            filename = os.fsdecode(file)
          
            # load x
            with open(f'./data/fp/binary/{filename}', 'r') as x_file:
                x_data = json.load(x_file)
                x = np.asarray(x_data)
            
            
            # divides data into 80% training and 20% external validation
            x_train, x_ext, y_train, y_ext = train_test_split(x, y, test_size=testSize, train_size=trainSize, random_state=randomState, stratify=y)



            with open(f'./data/curated_data/y/binary/y_train/y_train_{filename[:-5]}.json', 'w', encoding='utf-8') as f:
                json.dump(y_train, f, cls=NumpyArrayEncoder)
            with open(f'./data/curated_data/y/binary/y_ext/y_ext_{filename[:-5]}.json', 'w', encoding='utf-8') as f:
                json.dump(y_ext, f, cls=NumpyArrayEncoder)      
            with open(f'./data/curated_data/x/binary/x_ext/x_ext_{filename[:-5]}.json', 'w', encoding='utf-8') as f:
                json.dump(x_ext, f, cls=NumpyArrayEncoder)
            with open(f'./data/curated_data/x/binary/x_train/x_train_{filename[:-5]}.json', 'w', encoding='utf-8') as f:
                json.dump(x_train, f, cls=NumpyArrayEncoder)

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1], 'fn': cm[1, 0], 'tp': cm[1, 1]}

def bayessearch(models, search_spaces, scorer, outer_cv, inner_cv, x, y, fp, n_jobs=-1, random_state = 0):

    count = 0

    for index, model in enumerate(models):
        # invoke bayes search
        opt = BayesSearchCV(
            estimator=model, 
            search_spaces=search_spaces[index],
            scoring=scorer,
            cv=inner_cv,
            n_points=1,
            n_jobs=n_jobs,
            return_train_score=False,
            refit=False,
            optimizer_kwargs={'base_estimator': 'GP'},
            random_state=random_state
        )

        # callback funtions
        overdone_control = DeltaYStopper(delta=0.0001)
        time_limit_control = DeadlineStopper(total_time=60*60*4) #7h

        # search for best params
        best_params = report_perf(opt, x, y, callbacks=[overdone_control, time_limit_control])

        if count == 0:

            # Transferring best parameters to basic classifier
            basic_clas = xgb.XGBClassifier(random_state=0, booster='gbtree', **best_params)

            # cross validate with best params
            
            cv_results = cross_validate(basic_clas, x, y, cv=outer_cv, scoring=confusion_matrix_scorer)
            cv_auc = cross_validate(basic_clas, x, y, cv=outer_cv, scoring='roc_auc')
            cv_auc = pd.DataFrame(cv_auc)
            cv_auc.to_csv(f'./data/models/binary/metrics/xgb_classifier_auc_{fp}.csv', index = False)
            #training with entire dataset
            xgb_model = basic_clas.fit(x, y)

            # save model
            dump(xgb_model, f'./data/models/binary/xgb_classifier_{fp}.joblib')

            # save cross validation metrics
            cv_results = pd.DataFrame(cv_results)
            cv_results.to_csv(f'./data/models/binary/metrics/xgb_classifier_5cv_{fp}.csv', index = False)

            count += 1

        elif count == 1:

            # Transferring best parameters to basic classifier
            basic_clas = LGBMClassifier(random_state=0, **best_params)

            # cross validate with best params
            cv_results = cross_validate(basic_clas, x, y, cv=outer_cv, scoring=confusion_matrix_scorer)
            cv_auc = cross_validate(basic_clas, x, y, cv=outer_cv, scoring='roc_auc')
            cv_auc = pd.DataFrame(cv_auc)
            cv_auc.to_csv(f'./data/models/binary/metrics/lgbm_classifier_auc_{fp}.csv', index = False)


            #training with entire dataset
            xgb_model = basic_clas.fit(x, y)

            #save model
            dump(xgb_model, f'./data/models/binary/lgbm_classifier_{fp}.joblib')

            # save cross validation metrics
            cv_results = pd.DataFrame(cv_results)
            cv_results.to_csv(f'./data/models/binary/metrics/lgbm_classifier_5cv_{fp}.csv', index = False)

            count += 1

        elif count == 2:

            # Transferring best parameters to basic classifier
            basic_clas = AdaBoostClassifier(random_state=0, **best_params)

            # cross validate with best params
            cv_results = cross_validate(basic_clas, x, y, cv=outer_cv, scoring=confusion_matrix_scorer)
            cv_auc = cross_validate(basic_clas, x, y, cv=outer_cv, scoring='roc_auc')
            cv_auc = pd.DataFrame(cv_auc)
            cv_auc.to_csv(f'./data/models/binary/metrics/ada_classifier_auc_{fp}.csv', index = False)

            #training with entire dataset
            xgb_model = basic_clas.fit(x, y)

            # save model
            dump(xgb_model, f'./data/models/binary/ada_classifier_{fp}.joblib')

            # save cross validation metrics
            cv_results = pd.DataFrame(cv_results)
            cv_results.to_csv(f'./data/models/binary/metrics/ada_classifier_5cv_{fp}.csv', index = False)


def validatemodels ():

    # load model
    for clf in os.listdir('./data/models/binary/'):
        if clf.endswith('joblib'):

            # extract fingerprint name used
            s = clf
            result = re.search('classifier_(.*).joblib', s)
            fp_used = result.group(1)

            #extract 5cv metrics name
            result = re.search(f'(.*){fp_used}.joblib', s)
            modelname = result.group(1)
            
            # load model
            model = load(f'./data/models/binary/{clf}')


            with open(f'./data/curated_data/y/binary/y_ext/y_ext_{fp_used}.json', 'r') as y_ext:
                y_ext = json.load(y_ext)
                y_ext = np.asarray(y_ext)

            # load x_ext
            with open(f'./data/curated_data/x/binary/x_ext/x_ext_{fp_used}.json', 'r') as x_ext:
                x_ext = json.load(x_ext)
                x_ext = np.asarray(x_ext)

            # calculate external metrics
            metrics, cm = metrics_binary(model, x_ext, y_ext)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
            disp.plot()

            # save fig
            plt.savefig(f'./data/figs/confusion_matrix_{modelname}{fp_used}.png', dpi=300)


            A = [cm[1, 1], cm[0, 0]]
            B = [cm[1, 0], cm[0, 1]]

            fig = plt.figure(facecolor="white")

            ax = fig.subplots()
            bar_width = 0.9
            bar_l = np.arange(1, 3)
            tick_pos = [i + (bar_width / 30) for i in bar_l]

            ax2 = ax.bar(bar_l, A, width=bar_width, label="A", color="darkseagreen")
            ax1 = ax.bar(bar_l, B, bottom=A, width=bar_width, label="B", color="lightcoral")
            ax.set_ylabel("Count", fontsize=10)
            totals = [sum([cm[1, 1], cm[1, 0]]), sum([cm[0, 0], cm[0, 1]])]
            plt.legend(["Predicted correctly", "Predicted wrongly"], bbox_to_anchor=(0.45, -0.09), loc='upper center', ncol=1)
            plt.xticks(tick_pos, [f"Total Negatives: {totals[0]}", f"Total Positives: {totals[1]}"], fontsize=10)
            plt.title(f'Confusion Matrix {modelname}{fp_used}')
            plt.yticks(fontsize=10)

            for r1, r2 in zip(ax2, ax1):
                h1 = r1.get_height()
                h2 = r2.get_height()
                plt.text(r1.get_x() + r1.get_width() / 2., h1 / 2., "%d" % h1, ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")
                plt.text(r2.get_x() + r2.get_width() / 2., h1 + h2 / 2., "%d" % h2, ha="center", va="bottom", color="white", fontsize=10, fontweight="bold")

            plt.savefig(f'./data/figs/barplotconfusion_matrix_{modelname}{fp_used}.png', dpi=300, bbox_inches = 'tight')
