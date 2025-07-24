import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

def sample_distribution_compare(center_name, sampler, ori_minor, synthetic_samples):
    output_dir = f"{sampler}_{center_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    ks_results = []
    pdf_path = os.path.join(output_dir, f'{sampler}_{center_name}_distribution_validation.pdf')
    with PdfPages(pdf_path) as pdf:
        for feature in ori_minor.columns.tolist():
            orig_feature = ori_minor[feature]
            synth_feature = synthetic_samples[feature]

            ##
            plt.figure(figsize=(10, 6))
            plt.axis('off')
            sns.kdeplot(orig_feature, fill=True, alpha=0.5, label="Original", color="blue")
            sns.kdeplot(synth_feature, fill=True, alpha=0.5, label=f"{sampler}-Generated", color="orange")

            # KS-test
            ks_stat, p_value = stats.ks_2samp(orig_feature, synth_feature)
            plt.title(f"{sampler}_{center_name}_Distribution of Feature '{feature}'\nKS-test p-value={p_value:.4f}")
            plt.legend()
            pdf.savefig()
            # plt.show()
            plt.close()

            ks_results.append({
                'Center': center_name,
                'Feature': feature,
                'ks_stat': ks_stat,
                'p_value': p_value,
                'Significant': p_value < 0.05
            })
    ks_df = pd.DataFrame(ks_results)
    ks_df.to_csv(os.path.join(output_dir, f"{sampler}_{center_name}_Distributional Analysis.csv"), index=False)



