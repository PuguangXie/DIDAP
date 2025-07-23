import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from Sample_distribution_analysis import sample_distribution_compare


def augmentation(center_name, X_train, y_train, sampler, dis_flag, random_state=1):
    sampler_map = {
        'None': None,
        'smote': SMOTE(random_state=random_state),
        'ros': RandomOverSampler(random_state=random_state)
    }
    if sampler not in sampler_map:
        raise ValueError(f"Unknown Augmentation Methods")

    minor_indices = np.where(y_train == 1)[0]
    ori_minor = X_train.iloc[minor_indices]

    sampler_instance = sampler_map[sampler]

    if sampler == "None":
        return X_train, y_train

    elif sampler == "smote":
        X_resampled, y_resampled = sampler_instance.fit_resample(X_train, y_train)
        synthetic_indices = np.where((y_resampled == 1) & (np.isin(X_resampled, ori_minor).all(axis=1) == False))[0]
        synthetic_samples = X_resampled.iloc[synthetic_indices]
        # print(f"Original Distribution in {center_name}:", np.bincount(y_train))
        # print(f"Distribution after {sampler} Augmentation in {center_name}:", np.bincount(y_resampled))
        if dis_flag:
            sample_distribution_compare(center_name, sampler, ori_minor, synthetic_samples,random_state=random_state)
        return X_resampled, y_resampled

    elif sampler == "ros":
        X_resampled, y_resampled = sampler_instance.fit_resample(X_train, y_train)
        original_hashes = [hash(tuple(row)) for row in ori_minor.values]
        resampled_hashes = [hash(tuple(row)) for row in X_resampled.values]
        synthetic_indices = []
        for i, h in enumerate(resampled_hashes):
            if y_resampled[i] == 1:
                if h not in original_hashes or resampled_hashes.index(h) != i:
                    synthetic_indices.append(i)
        synthetic_samples = X_resampled.iloc[synthetic_indices]
        # print(f"Original Distribution in {center_name}:", np.bincount(y_train))
        # print(f"Distribution after {sampler} Augmentation in {center_name}:", np.bincount(y_resampled))
        if dis_flag:
            sample_distribution_compare(center_name, sampler, ori_minor, synthetic_samples,random_state=random_state)
        return X_resampled, y_resampled


