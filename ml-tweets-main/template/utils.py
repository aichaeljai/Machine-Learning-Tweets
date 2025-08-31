import pandas as pd
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import spacy
import contractions
import re
import emoji
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler,  MultiLabelBinarizer
from sklearn.metrics import silhouette_score, make_scorer, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, multilabel_confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore

nlp = spacy.load("en_core_web_sm")
rs = 42

## Dataset modification ##

def set_target(df, classif_mode):
    if classif_mode == 0:
        df['target'] = df.apply(lambda x: [int(x['science_related'])], axis=1)
        class_map = {
            0: "not science related",
            1: "science related",
        }
    elif classif_mode == 1:
        df = df[df['science_related'] == 1]
        df['target'] = df.apply(lambda x: [int(x['scientific_claim']) | int(x['scientific_reference']), int(x['scientific_context'])], axis=1)
        class_map = {
            0: "knowledge",
            1: "context",
        }
    elif classif_mode == 2:
        df = df[df['science_related'] == 1]
        df['target'] = df.apply(lambda x: [int(x['scientific_claim']), int(x['scientific_reference']), int(x['scientific_context'])], axis=1)
        class_map = {
            0: "claim",
            1: "reference",
            2: "context"
        }
    else:
        raise ValueError(f"classif_mode {classif_mode} is not supported, choose within 0, 1, 2")
    
    df = df.drop(columns=['tweet_id', 'science_related', 'scientific_claim', 'scientific_reference', 'scientific_context'])   
    return df, class_map

def over_under_sample(class_id, class_value, n, df):
    # set the number of data with class <class_id> set to <class_value> to <n> data
    # perform undersampling if the current number of data with <class_id> is superior to <n>, perform undersampling otherwise
    sample_from = df[df['target'].apply(lambda x: x[class_id] == class_value)]
    not_sample_from = df[df['target'].apply(lambda x: x[class_id] != class_value)]

    if len(sample_from) < n: # Oversampling
        to_sample = n - len(sample_from)
        new_data = resample(
            sample_from, 
            replace=len(sample_from) < to_sample,
            n_samples=to_sample,  
            random_state=rs
        )
        resample_from = pd.concat([sample_from, new_data])
    
    elif len(sample_from) > n: # Undersampling
        to_sample = n
        new_data = resample(
            sample_from, 
            replace=False,
            n_samples=to_sample,  
            random_state=rs
        )
        resample_from = new_data
    
    else:
        resample_from = sample_from
    
    new_df = pd.concat([resample_from, not_sample_from])    
    new_df = new_df.sample(frac=1, random_state=rs).reset_index(drop=True)
    return new_df

def vectorize_df(df, vectorizer):
    vectorizer.fit(df["text"])
    vectorized = pd.DataFrame(
        data=vectorizer.transform(df['text']).toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    vectorized["target"] = df["target"].reset_index(drop=True)
    return vectorized

def normalize_df(df, scaler):
    scaled_array = scaler.fit_transform(df.drop(columns=["target"]))
    scaled = pd.DataFrame(scaled_array, columns=df.drop(columns=["target"]).columns)
    scaled["target"] = df["target"].reset_index(drop=True)
    return scaled


## PLOTS ##

def class_balance_plot(df, class_map):
    classes_counts = df.copy()
    classes_counts['t_index'] = classes_counts['target'].apply(lambda x: list(enumerate(x)))
    classes_counts = classes_counts.explode(['target', 't_index'], ignore_index=True)
    classes_counts['classe'] = classes_counts['t_index'].apply(lambda x: f"{x[0]}={class_map[x[0]]}")

    sns.countplot(data=classes_counts, x="classe", hue="target")
    plt.title("Répartition des donnees dans chaque classe")
    plt.show()


def display_separation(data, class_map):
    # PCA projection
    pca = PCA(n_components=3)
    features = data.drop(columns=["target"])
    reduced_data = pca.fit_transform(features)
    reduced_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2', 'PC3'])

    # Handle both single-label and multi-label targets
    y_raw = data["target"].values
    is_multilabel = isinstance(y_raw[0], (list, np.ndarray)) and len(y_raw[0]) > 1

    if is_multilabel:
        # Convert to string labels for plotting
        label_strs = [" & ".join([class_map[i] if v else "NOT " + class_map[i] for i, v in enumerate(y)]) for y in y_raw]
        le = LabelEncoder()
        reduced_df['target'] = le.fit_transform(label_strs)
        target_names = le.classes_
    else:
        labels = [class_map[y[0]] for y in y_raw]
        le = LabelEncoder()
        reduced_df['target'] = le.fit_transform(labels)
        target_names = le.classes_

    # Get consistent color palette
    num_classes = len(np.unique(reduced_df['target']))
    palette = sns.color_palette("husl", num_classes)
    colors = [palette[i] for i in reduced_df['target']]

    # Create subplots
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # 3D plot
    scatter = ax1.scatter(
        reduced_df['PC1'], reduced_df['PC2'], reduced_df['PC3'],
        c=colors, marker='o'
    )
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')

    # Manual legend for 3D
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=target_names[i],
               markerfacecolor=palette[i], markersize=10)
        for i in range(num_classes)
    ]
    ax1.legend(handles=legend_elements, title="Classes", loc='best')

    # 2D plot using seaborn (consistent color mapping)
    sns.scatterplot(
        x='PC1', y='PC2',
        hue=reduced_df['target'],
        palette=palette,
        data=reduced_df,
        ax=ax2,
        legend='full'
    )
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.legend(title="Classes", labels=target_names)

    plt.suptitle("Data Separation with PCA Reduction (3D / 2D)", fontsize=14)
    plt.tight_layout()
    plt.show()


# Draw a boxplot for each metrics
def plot_model_comparison(results_dict, metric):
    if metric not in ['accuracy', 'f1']:
        raise ValueError("Metric must be either 'accuracy' or 'f1'.")

    # Flatten the results into a DataFrame for seaborn
    records = []
    for name, metrics in results_dict.items():
        if metric not in metrics:
            raise ValueError(f"Metric '{metric}' not found in results for model: {name}")
        for score in metrics[metric]:
            records.append({'Model': name, metric.capitalize(): score})

    df_plot = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_plot, x='Model', y=metric.capitalize(), hue='Model', palette="Set2", dodge=False)
    plt.title(f'Model Comparison by {metric.capitalize()}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def display_confusions(conf_matrices, class_map):
    num_classes = conf_matrices.shape[0]
    cols = 3  # number of columns in the subplot grid
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))

    for i, cm in enumerate(conf_matrices):
        row, col = divmod(i, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_title(f'Class {class_map[i]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    # Hide any unused subplots
    for j in range(i + 1, rows * cols):
        row, col = divmod(j, cols)
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def plot_hs_results2(df, param_name, metric_name='mean_test_score'):
    # Ensure no removal of data, we preserve all including None/NaN
    
    # Check if all values are numeric and not NaN, only if so we proceed with sorting
    if df[f'param_{param_name}'].apply(pd.to_numeric, errors='coerce').notna().all():
        # If all values are numeric and valid, sort the dataframe by the hyperparameter values
        df = df.sort_values(by=f'param_{param_name}')
    else:
        df['param_max_depth'] = df['param_max_depth'].apply(lambda x: str(x))
    
    # Get unique values of the hyperparameter (for proper ordering)
    unique_values = df[f'param_{param_name}'].unique()

    # If the hyperparameter is numeric, sort the unique values
    if pd.api.types.is_numeric_dtype(df[f'param_{param_name}']):
        unique_values = sorted(unique_values)

    # Create a figure for plotting
    plt.figure(figsize=(12, 6))

    # Plot Accuracy vs Hyperparameter
    sns.boxplot(x=f'param_{param_name}', y=metric_name, data=df, order=unique_values, color='blue')
    plt.title(f'Accuracy vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')

    # Display the plot
    plt.tight_layout()
    plt.show()




## Metrics ##

def get_silouhette(df):
    X = df.drop(columns=["target"])
    y = np.array(df["target"].to_list())

    # Calculate silhouette score for each label (per label)
    sil_scores_multi = []
    for label_idx in range(y.shape[1]):
        sil_score = silhouette_score(X, y[:, label_idx])
        sil_scores_multi.append(sil_score)

    # Average silhouette score for multi-labeled data
    avg_sil_score = sum(sil_scores_multi) / len(sil_scores_multi)
    return avg_sil_score

def multivariate_zscore(df):
    # Compute Z-scores only for numeric columns
    numeric_data = df.select_dtypes(include='number')
    z_scores = zscore(numeric_data)

    # Compute the Euclidean norm (L2) of Z-scores for each row
    row_norms = np.linalg.norm(z_scores, axis=1)

    return row_norms

def get_scaling_sum(df, scalers):
    results = []
    results.append(("none", get_silouhette(df)))
    for s in scalers.keys():
        scaled = normalize_df(df, scalers[s])
        results.append((s, get_silouhette(scaled)))

    results = sorted(results, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(results, columns=['Scaling', 'Silouhette'])

# Compute an average accuracy accross labels
def multilabel_accuracy(y_true, y_pred):
    return np.mean([accuracy_score(y_true[:, i], y_pred[:, i]) 
                    for i in range(y_true.shape[1])])

# Compute an average f1-score accross labels
def multilabel_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='samples')



## Text Processing ##

def encode(txt, s):
    def encode_tags(txt, setting):
        if setting == "replace":
            return re.sub(r'#([\w_]+)', r' hashtag \1 ', txt)
        elif setting == "delete":
            return re.sub(r'#([\w_]+)', r' ', txt)
        return txt

    def encode_mention(txt, setting):
        if setting == "replace":
            return re.sub(r'@([\w_]+)', r' mention \1 ', txt)
        elif setting == "delete":
            return re.sub(r'@([\w_]+)', r' ', txt)
        return txt

    def encode_url(txt, setting):
        expr = r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+'
        if setting == "encode":
            return re.sub(expr, " url ", txt)
        elif setting == "delete":
            return re.sub(expr, " ", txt)
        return txt

    def encode_underscore(txt, setting):
        if setting == "encode" or setting == "delete":
            return txt.replace("_", " ")
        return txt

    def encode_emoji(txt, setting):
        if setting == "replace":
            replacement = " emoticon "
        elif setting == "remove":
            replacement = " "
        else:
            return txt
        
        new_txt = txt
        for token in reversed(nlp(txt)):
            if emoji.is_emoji(token.text):
                new_txt = new_txt[:token.idx] + replacement + new_txt[token.idx + len(token.text):]
                
        return new_txt

    def encode_numbers(txt, setting):
        if setting == "none":
            return txt
        
        doc = nlp(txt)
        new_txt = txt
        for ent in reversed(doc.ents):  # reverse so replacements don't mess up indices
            if not "#" in ent.text and ent.label_ in ["CARDINAL", "ORDINAL", "PERCENT", "MONEY"] :
                if setting == "replace":
                    replacement = " number " + ent.label_.lower() + " "
                else:
                    replacement = " "
                new_txt = new_txt[:ent.start_char] + replacement + new_txt[ent.end_char:]

        return new_txt
    new_txt = encode_tags(txt, s["tags"])
    new_txt = encode_mention(new_txt, s["mention"])
    new_txt = encode_url(new_txt, s["url"])
    new_txt = encode_emoji(new_txt, s["emoji"])
    new_txt = encode_numbers(new_txt, s["numbers"])
    new_txt = encode_underscore(new_txt, s["underscore"])
    return new_txt

def filter_words(txt, f):
    new_txt = []
    for t in nlp(txt):
        if t.is_alpha and t.pos_ in f:
            new_txt.append(t.text)
    return " ".join(new_txt)


## Models

# Perform cross validation for 1 model return f1-score and accuracy for each fold
def cross_validate_model(dataframe, model, n_splits=5, random_state=None):
    X = dataframe.drop(columns=["target"]).values
    y = np.array(list(dataframe["target"].values))

    is_multilabel = y.shape[1] > 1

    # Define custom scorers
    if is_multilabel:
        scoring = {
            'accuracy': make_scorer(multilabel_accuracy),
            'f1': make_scorer(multilabel_f1)
        }
        model = MultiOutputClassifier(model)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    else:
        y = dataframe["target"].apply(lambda x: x[0]).values
        scoring = ['accuracy', 'f1']
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Run cross-validation
    results = cross_validate(
        model, X, y, 
        scoring=scoring, 
        cv=cv, 
        return_train_score=False,
        n_jobs=-1
    )

    return {
        'accuracy': results['test_accuracy'],
        'f1': results['test_f1']
    }

# Evaluate multiple model on the same datas
def evaluate_by_models(dataframe, models, n_splits=5, random_state=None):
    results = {}

    for name, model in models.items():
        #print(f"Evaluating model: {name}")
        metrics = cross_validate_model(
            dataframe=dataframe,
            model=model,
            n_splits=n_splits,
            random_state=random_state
        )
        results[name] = metrics

    return results

# Evaluate one model on multiple data
def evaluate_by_df(dataframes, model, n_splits=5, random_state=None):
    results = {}

    for name, df in dataframes.items():
        print(f"Evaluating data: {name}")
        metrics = cross_validate_model(
            dataframe=df,
            model=model,
            n_splits=n_splits,
            random_state=random_state
        )
        results[name] = metrics

    return results

def hyperparam_search(df, model, params, metric='accuracy', n_splits=5, random_state=None):
    X = df.drop(columns=["target"]).values
    y = np.array(list(df["target"].values))

    is_multilabel = y.shape[1] > 1

    if is_multilabel:
        params = {f'estimator__{k}': v for k, v in params.items()}
        scoring = {
            'accuracy': make_scorer(multilabel_accuracy),
            'f1': make_scorer(multilabel_f1)
        }
        scoring = scoring[metric]
        model = MultiOutputClassifier(model)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    else:
        y = df["target"].apply(lambda x: x[0]).values
        scoring = metric
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring=scoring, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    return grid_search

def hyperparam_rnd_search(df, model, params, n_iter=100, metric='accuracy', n_splits=5, random_state=None):
    X = df.drop(columns=["target"]).values
    y = np.array(list(df["target"].values))

    is_multilabel = y.shape[1] > 1

    if is_multilabel:
        params = {f'estimator__{k}': v for k, v in params.items()}
        scoring = {
            'accuracy': make_scorer(multilabel_accuracy),
            'f1': make_scorer(multilabel_f1)
        }
        scoring = scoring[metric]
        model = MultiOutputClassifier(model)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    else:
        y = df["target"].apply(lambda x: x[0]).values
        scoring = metric
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,             
        random_state=random_state
    )
    random_search.fit(X, y)

    return random_search

def evaluate_model(df, model, n_splits=5, random_state=None):
    X = df.drop(columns=["target"]).values
    y = np.array(list(df["target"].values))

    is_multilabel = y.shape[1] > 1

    if is_multilabel:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        average_type = 'macro'
    else:
        y = df["target"].apply(lambda x: x[0]).values
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        average_type = 'binary'

    y_pred = cross_val_predict(model, X, y, cv=cv)

    report = classification_report(y, y_pred)

    cv_results = cross_validate(model, X, y, cv=cv, scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

    # Extract the results for each metric
    accuracies = cv_results['test_accuracy']
    precision = cv_results['test_precision_macro']
    recall = cv_results['test_recall_macro']
    f1 = cv_results['test_f1_macro']

    if not is_multilabel:
        cm = np.array([confusion_matrix(y, y_pred)])
    else:
        cm = multilabel_confusion_matrix(y, y_pred)

    return {
        "report": report, 
        "acc": accuracies,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm
    }   


## Misc ##

def get_data_text(df, index):
    res = []
    for col in df.drop(columns=["target"]).columns:
        if df[col].iloc[index] > df[col].mean():
            res.append(col)
    return " ".join(res)