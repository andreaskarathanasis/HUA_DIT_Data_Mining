from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_roc_curves(results, model):
    sns.reset_orig()
    sns.set_theme()
    plt.figure(figsize=(8, 6))    

    for sampling in results[model]:
        fpr, tpr, thr = roc_curve(results[model][sampling]['true'], results[model][sampling]['preds'])
        plt.plot(fpr, tpr, label=f'{sampling} AUC: {auc(fpr, tpr):.2}')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    
    plt.title(f"{model} ROC curves")
    plt.legend()
    plt.show()


def plot_classification_report(reports, name):
    sns.reset_orig()
    sns.set_theme()
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    sns.heatmap(pd.DataFrame(reports[name]['default']['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True).set_title('default')
    sns.heatmap(pd.DataFrame(reports[name]['upsampled']['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True).set_title('upsampled')
    sns.heatmap(pd.DataFrame(reports[name]['downsampled']['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True).set_title('downsampled')
    sns.heatmap(pd.DataFrame(reports[name]['SMOTE']['report']).iloc[:-1, :].T, annot=True, vmin=0, vmax=1, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True).set_title('SMOTE')
    plt.suptitle(f'{name} classification report')
    plt.show()


def plot_confusion_matrices(results, model):
    cf_matrices = [confusion_matrix(results[model][sampling]['true'], results[model][sampling]['preds']) for sampling in results[model]]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))

    sns.reset_orig()
    sns.set_theme()

    labels = [name for name in results[model].keys()]

    sns.heatmap(pd.DataFrame(cf_matrices[0]), fmt='g', annot=True, ax=axes[0][0], xticklabels=False, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[0])
    sns.heatmap(pd.DataFrame(cf_matrices[1]), fmt='g', annot=True, ax=axes[0][1], xticklabels=False, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[1])
    sns.heatmap(pd.DataFrame(cf_matrices[2]), fmt='g', annot=True, ax=axes[1][0], xticklabels=True, yticklabels=True, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[2])
    sns.heatmap(pd.DataFrame(cf_matrices[3]), fmt='g', annot=True, ax=axes[1][1], xticklabels=True, yticklabels=False, cbar=True, linewidths=0.5, linecolor='black').set_title(labels[3])
    
    plt.suptitle(f'{model} - Confusion Matrices')
    plt.show()