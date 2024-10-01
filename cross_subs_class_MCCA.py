#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.utils import shuffle
import time
from mne_features.feature_extraction import FeatureExtractor
import seaborn as sns
from MCCA_main.MCCA import MCCA

def magnified(values, factor):
    """Magnify the signal values for better visualization."""
    return values * factor

def plot_aligned_channels_comparison(df, subject_id, fnirs_columns, title="fNIRS Channels"):
    """Compare the fNIRS channels for Low and High Load conditions for a specific subject."""
    low_load_data = df[(df['ID'] == subject_id) & (df['load_label'] == 'Low Load')]
    high_load_data = df[(df['ID'] == subject_id) & (df['load_label'] == 'High Load')]

    low_load_data[fnirs_columns] = low_load_data[fnirs_columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    high_load_data[fnirs_columns] = high_load_data[fnirs_columns].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    plt.figure(figsize=(12, 8))
    time_low, time_high = low_load_data['time'].values, high_load_data['time'].values
    offset = 10
    magnification_factor = 2
    
    for i, channel in enumerate(fnirs_columns):
        plt.plot(time_low, magnified(low_load_data[channel].values, magnification_factor) + i * offset, 
                 color='blue', label=f'Low Load' if i == 0 else "")
        plt.plot(time_high, magnified(high_load_data[channel].values, magnification_factor) + i * offset, 
                 color='red', label=f'High Load' if i == 0 else "")
    
    plt.title(f"{title} - Subject {subject_id} (Low vs High Load Comparison)")
    plt.xlabel('Time (s)')
    plt.ylabel('Activity (Magnified and Offset)')
    plt.legend(loc='upper right', fontsize='small', bbox_to_anchor=(1.15, 1))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def check_labels(df_hbr, all_labels):
    """Check if the labels in all_labels correspond to 'load_label' in df_hbr."""
    is_correct = True
    for subject_idx, subject_id in enumerate(df_hbr['ID'].unique()):
        subject_data = df_hbr[df_hbr['ID'] == subject_id]
        subject_labels = all_labels[subject_idx]
        time_per_trial = subject_data['time'].nunique()
        
        for i in range(len(subject_labels)):
            start_idx = i * time_per_trial
            df_label = subject_data['load_label'].iloc[start_idx]
            df_label_converted = 0 if df_label == 'Low Load' else 1
            if subject_labels[i] != df_label_converted:
                print(f"Mismatch for subject {subject_id} at trial {i + 1}: Expected {df_label_converted}, got {subject_labels[i]}")
                is_correct = False
    
    print("All labels match correctly." if is_correct else "There were mismatches in the labels.")

def prepare_and_apply_mcca(X_train_subjects, Y_train_subjects, n_pcs, n_ccs, regularization_strength):
    """Prepare training data and apply MCCA."""
    low_load_trials, high_load_trials = [], []
    for trials, labels in zip(X_train_subjects, Y_train_subjects):
        low_trials, high_trials = trials[labels == 0], trials[labels == 1]
        low_load_trials.append(np.mean(low_trials, axis=0))
        high_load_trials.append(np.mean(high_trials, axis=0))
    
    subjects_train_matrix = np.concatenate([low_load_trials, high_load_trials], axis=1)
    mcca = MCCA(n_components_pca=n_pcs, n_components_mcca=n_ccs, r=regularization_strength)
    mcca_results = mcca.obtain_mcca(subjects_train_matrix)
    return mcca, mcca_results

def mcca_to_feature(transformed_train):
    """Transform MCCA data to feature-ready format."""
    flattened_trials = [trial for subject_trials in transformed_train for trial in subject_trials]
    X_transformed = np.array(flattened_trials)
    return X_transformed.reshape(X_transformed.shape[0], X_transformed.shape[2], X_transformed.shape[1])

# Data preprocessing
df = pd.read_csv(r'C:\Users\njeuk\OneDrive\Documents\MCCA\Dataset\df_epochs.csv', delimiter=';')
df['time'] = pd.to_numeric(df['time'].str.replace(',', '.'), errors='coerce')
fnirs_columns_hbr = [col for col in df.columns if 'hbo' in col or 'hbr' in col]
for col in fnirs_columns_hbr:
    df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

# Filter out unnecessary conditions
df_filtered = df[~df['condition'].isin(['Rest', 'LowSil', 'HighSil', 'Baseline'])]
df_filtered['load_type'] = df_filtered['condition'].apply(lambda x: 'High Load' if 'High' in x else 'Low Load')
df_hbr = df_filtered[fnirs_columns_hbr + ['time', 'condition', 'epoch', 'ID']]

# Create a new column 'load_label'
high_load_conditions = ['HighPos', 'HighNeg', 'HighNeu']
low_load_conditions = ['LowPos', 'LowNeg', 'LowNeu']
df_hbr['load_label'] = df_hbr['condition'].apply(lambda x: 'High Load' if x in high_load_conditions else 'Low Load')

def extract_trials_from_subject(df_hbr, subject_id, fnirs_columns_hbr):
    """Extract trials and labels for each subject."""
    subject_data = df_hbr[df_hbr['ID'] == subject_id]
    time_per_trial = subject_data['time'].nunique()
    n_sensors = len(fnirs_columns_hbr)
    n_trials = len(subject_data) // time_per_trial

    trial_matrix = np.zeros((n_trials, time_per_trial, n_sensors))
    labels = np.zeros(n_trials)
    
    for i in range(n_trials):
        start_idx = i * time_per_trial
        end_idx = (i + 1) * time_per_trial
        trial_matrix[i] = subject_data[fnirs_columns_hbr].iloc[start_idx:end_idx].values
        labels[i] = 0 if subject_data['load_label'].iloc[start_idx] == 'Low Load' else 1
    
    return trial_matrix, labels

# Extract trials and labels from all subjects
all_trials, all_labels = [], []
for subject_id in df_hbr['ID'].unique():
    trials, labels = extract_trials_from_subject(df_hbr, subject_id, fnirs_columns_hbr)
    all_trials.append(trials)
    all_labels.append(labels)

X, Y = all_trials, all_labels

# Decoding using MCCA and SVM
fe = FeatureExtractor(sfreq=5.81, selected_funcs=['mean', 'std', 'ptp_amp', 'skewness', 'kurtosis'])
loo = LeaveOneOut()
subject_ids = np.arange(len(X))
f1_scores_mcca, f1_scores_no_mcca = [], []
confusion_matrix_mcca = np.zeros((2, 2))
confusion_matrix_no_mcca = np.zeros((2, 2))
use_mcca_list = [True, False]

for use_mcca in use_mcca_list:
    f1_scores, overall_confusion_matrix = [], np.zeros((2, 2))
    for fold, (train_idx, test_idx) in enumerate(loo.split(subject_ids)):
        start_time = time.time()

        X_train_subjects = [X[i] for i in train_idx]
        Y_train_subjects = [Y[i] for i in train_idx]
        X_test_subject, Y_test_subject = X[test_idx[0]], Y[test_idx[0]]
        
        if use_mcca:
            mcca, mcca_results = prepare_and_apply_mcca(X_train_subjects, Y_train_subjects, n_pcs=41, n_ccs=10, regularization_strength=0)
            transformed_train = [mcca.transform_trials(trials, subject=i) for i, trials in enumerate(X_train_subjects)]
            X_transformed_train = mcca_to_feature(transformed_train)
            X_transformed_test = np.transpose(mcca.transform_trials(X_test_subject), (0, 2, 1))
        else:
            X_transformed_train = np.concatenate([subj_data.transpose(0, 2, 1) for subj_data in X_train_subjects], axis=0)
            X_transformed_test = X_test_subject.transpose(0, 2, 1)

        X_features_train, X_features_test = fe.fit_transform(X_transformed_train), fe.transform(X_transformed_test)
        X_features_train, Y_train = shuffle(X_features_train, np.hstack(Y_train_subjects))

        clf = SVC(kernel='linear', C=0.1, max_iter=5000)
        clf.fit(X_features_train, Y_train)
        Y_pred = clf.predict(X_features_test)

        fold_f1_score = f1_score(Y_test_subject, Y_pred, average='weighted')
        f1_scores.append(fold_f1_score)
        overall_confusion_matrix += confusion_matrix(Y_test_subject, Y_pred)

        print(f"Fold {test_idx[0] + 1} (MCCA = {use_mcca}): F1 Score = {fold_f1_score:.2f}, Time = {time.time() - start_time:.4f} sec")
    
    if use_mcca:
        f1_scores_mcca = f1_scores
        confusion_matrix_mcca = overall_confusion_matrix
    else:
        f1_scores_no_mcca = f1_scores
        confusion_matrix_no_mcca = overall_confusion_matrix


# %% PLOTTING AND PRINTING
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

def print_classification_results(f1_scores_mcca, f1_scores_no_mcca, confusion_matrix_mcca, confusion_matrix_no_mcca):
    # Calculate mean F1 scores
    mean_f1_mcca = np.mean(f1_scores_mcca)
    mean_f1_no_mcca = np.mean(f1_scores_no_mcca)

    # Print table header
    print("\n{:<20} {:<15} {:<15}".format("Metric", "With MCCA", "Without MCCA"))
    print("-" * 50)

    # Print F1 scores
    print("{:<20} {:<15.4f} {:<15.4f}".format("Mean F1 Score", mean_f1_mcca, mean_f1_no_mcca))

    # Calculate overall percentages based on the total number of predictions (for both cases)
    total_mcca = confusion_matrix_mcca.sum()
    total_no_mcca = confusion_matrix_no_mcca.sum()

    # Normalize confusion matrix to get percentage of each quadrant
    tn_mcca, fp_mcca, fn_mcca, tp_mcca = confusion_matrix_mcca.ravel() / total_mcca
    tn_no_mcca, fp_no_mcca, fn_no_mcca, tp_no_mcca = confusion_matrix_no_mcca.ravel() / total_no_mcca

    # Print normalized results (as percentages)
    print("{:<20} {:<15.2%} {:<15.2%}".format("True Positive (TP)", tp_mcca, tp_no_mcca))
    print("{:<20} {:<15.2%} {:<15.2%}".format("False Positive (FP)", fp_mcca, fp_no_mcca))
    print("{:<20} {:<15.2%} {:<15.2%}".format("True Negative (TN)", tn_mcca, tn_no_mcca))
    print("{:<20} {:<15.2%} {:<15.2%}".format("False Negative (FN)", fn_mcca, fn_no_mcca))

# Function to plot confusion matrix with custom layout
def plot_custom_confusion_matrix(tp, fp, tn, fn, title):
    # Create confusion matrix layout
    confusion_matrix_true = np.array([[tp, 0], [tn, 0]])  # True values (TP/TN)
    confusion_matrix_false = np.array([[0, fp], [0, fn]])  # False values (FP/FN)

    # Sum for color scaling
    tp_tn_sum = tp + tn  # Correct classifications (TP + TN)
    fp_fn_sum = fp + fn  # Incorrect classifications (FP + FN)

    # Normalize values to percentages
    tp_percent = tp * 100
    fp_percent = fp * 100
    tn_percent = tn * 100
    fn_percent = fn * 100

    # Create a grid of labels based on TP, FP, TN, FN percentages
    labels = np.array([[f'TP: {tp_percent:.2f}%', f'FP: {fp_percent:.2f}%'],
                       [f'TN: {tn_percent:.2f}%', f'FN: {fn_percent:.2f}%']])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot heatmap for True Positives/Negatives
    sns.heatmap(confusion_matrix_true, annot=labels, fmt='', cmap='RdYlGn', linewidths=0.5, square=True,
                vmin=0, vmax=tp_tn_sum, cbar_kws={"label": "True (%)"}, ax=ax, cbar=True, alpha=0.9)

    # Overlay heatmap for False Positives/Negatives with inverted scale
    sns.heatmap(confusion_matrix_false, annot=labels, fmt='', cmap='RdYlGn_r', linewidths=0.5, square=True,
                vmin=fp_fn_sum, vmax=0, cbar_kws={"label": "False (%)"}, ax=ax, cbar=True, alpha=0.9)

    # Set title and axis labels
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Adjust the axis tick labels to match the classification task
    ax.set_xticklabels(['High Load', 'Low Load'])
    ax.set_yticklabels(['High Load', 'Low Load'])

    # Adjust the layout to fit color bars
    plt.tight_layout()
    plt.show()


# Normalize the confusion matrix for MCCA and no MCCA cases
total_mcca = confusion_matrix_mcca.sum()
total_no_mcca = confusion_matrix_no_mcca.sum()

# Normalize the confusion matrix to percentages
tp_mcca, fp_mcca, tn_mcca, fn_mcca = confusion_matrix_mcca[0, 0] / total_mcca, confusion_matrix_mcca[0, 1] / total_mcca, confusion_matrix_mcca[1, 0] / total_mcca, confusion_matrix_mcca[1, 1] / total_mcca
tp_no_mcca, fp_no_mcca, tn_no_mcca, fn_no_mcca = confusion_matrix_no_mcca[0, 0] / total_no_mcca, confusion_matrix_no_mcca[0, 1] / total_no_mcca, confusion_matrix_no_mcca[1, 0] / total_no_mcca, confusion_matrix_no_mcca[1, 1] / total_no_mcca

# Plot confusion matrix with MCCA
plot_custom_confusion_matrix(tp_mcca, fp_mcca, tn_mcca, fn_mcca, 
                             "Confusion Matrix: With MCCA")

# Plot confusion matrix without MCCA
plot_custom_confusion_matrix(tp_no_mcca, fp_no_mcca, tn_no_mcca, fn_no_mcca, 
                             "Confusion Matrix: Without MCCA")



# %%
