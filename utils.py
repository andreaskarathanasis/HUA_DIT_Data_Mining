from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE


def build_resampled_datasets(dataset, test_ratio=0.3):
    # Separate testing split from dataset
    df_train, df_test = train_test_split(dataset, test_size=test_ratio, stratify=dataset['oscar_winners'])

    # Separate classes
    df_majority = df_train[(df_train['oscar_winners']==0)].reset_index().drop('index', axis=1)
    df_minority = df_train[(df_train['oscar_winners']==1)].reset_index().drop('index', axis=1)

    # Upsample the minority class of the dataset
    df_minority_upsampled = resample(df_minority, replace=True, n_samples= len(df_majority), random_state=42)
    df_minority_upsampled = df_minority_upsampled.reset_index().drop('index', axis=1)
    df_upsampled = pd.concat([df_minority_upsampled, df_majority]).sort_index(kind='merge')
    df_upsampled = df_upsampled.reset_index().drop('index', axis=1)

    # Downsample the majority class of the dataset
    df_majority_downsampled = resample(df_majority, replace=True, n_samples= len(df_minority), random_state=42)
    df_majority_downsampled = df_majority_downsampled.reset_index().drop('index', axis=1)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority]).sort_index(kind='merge')
    df_downsampled = df_downsampled.reset_index().drop('index', axis=1)

    # Resample using SMOTE method
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    df_smote_X, df_smote_Y = sm.fit_resample(df_train.drop('oscar_winners', axis=1), df_train['oscar_winners'])
    df_smote = pd.concat([pd.DataFrame(df_smote_X), pd.DataFrame(df_smote_Y)], axis=1)
    
    # Dataframes are returned
    return {
        'default' : dataset,
        'upsampled' : df_upsampled,
        'downsampled' : df_downsampled,
        'SMOTE' : df_smote
    }, df_test


def best_f1_score(results, model):
    best = results[model]['default']
    name = 'default'
    for key in results[model]:
        if results[model][key]['report']['1']['f1-score'] > best['report']['1']['f1-score']:
            best = results[model][key]
            name = key

    return best, name


def cluster_scatter_plot_2D(dataframe, clusters_column, xfeature, yfeature, title):
    no_oscars = dataframe[dataframe['oscar_winners'] == 0]
    oscars = dataframe[dataframe['oscar_winners'] == 1]
    
    colors_no = ('darkolivegreen', 'teal', 'purple', 'darkgoldenrod')
    colors_os = ('lime', 'cyan', 'magenta', 'yellow')

    # n_clusters = list(no_oscars[clusters_column].unique())

    clusters = list(no_oscars[clusters_column].unique())
    clusters.sort()

    plt.figure(figsize=(10,6))

    for cluster in clusters:
        plt.scatter(x=no_oscars[no_oscars[clusters_column]==cluster][xfeature], y=no_oscars[no_oscars[clusters_column]==cluster][yfeature], 
                    edgecolors='white', linewidth=0.3, s=50, color=colors_no[cluster], label=f'Cluster {cluster}')
    for cluster in clusters:
        plt.scatter(x=oscars[oscars[clusters_column]==cluster][xfeature], y=oscars[oscars[clusters_column]==cluster][yfeature], 
                    edgecolors='red', linewidth=0.5, s=80, color=colors_os[cluster], marker='^', label=f'Cluster {cluster} - oscar')

    plt.title(title)
    plt.xlabel(xfeature)
    plt.ylabel(yfeature)
    plt.legend()
    plt.show()