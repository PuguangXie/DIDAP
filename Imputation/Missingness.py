import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import missingno as msno
import pandas as pd


def missing(centers):
    results = []
    for center_name, center_data in centers.items():
        # print(center_data.shape[0])
        all_features = center_data.columns
        # Number and percentage of missningness
        miss_counts = center_data.isnull().sum()
        miss_percent = ((miss_counts/center_data.shape[0]) * 100).round(2)
        # print(miss_counts)
        # print(miss_percent)
        row_data = {
            'Center': center_name,
            'Total Samples': len(center_data)
        }
        for feature in center_data.columns:
            row_data[feature] = f"{miss_counts[feature]}\n({miss_percent[feature]}%)"

        results.append(row_data)
    df = pd.DataFrame(results)
    df.to_csv(f"Missingness Abstract.csv")
    return results


def visualize_missingness(centers, output_dir="missingness_visualization"):
    os.makedirs(output_dir, exist_ok=True)
    bia = {
    'DXH': 0.22,
    'FAH': 0.12
    }
    for center_name, center_data in centers.items():
        print(f"\nCreating visualization for {center_name} center...")

        miss_counts = center_data.isnull().sum()
        miss_percent = (center_data.isnull().mean() * 100).round(2)
        viz_df = pd.DataFrame({
            'Feature': miss_counts.index,
            'Missing Count': miss_counts.values,
            'Missing %': miss_percent.values
        })
        viz_df = viz_df.sort_values('Missing %', ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        # fig_height = max(8, 0.5 * len(viz_df))
        # fig_width = 14
        # fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(top=0.98, bottom=0.1, left=0.3, right=0.95)

        ax = sns.barplot(
            x='Missing %',
            y='Feature',
            data=viz_df,
            color='skyblue',
            orient='h'
        )
        for i, (count, percent) in enumerate(zip(viz_df['Missing Count'], viz_df['Missing %'])):
            # print(i)
            if percent == 0:
                label_x = bia[center_name]
                # ha = 'left'
            else:
                label_x = percent + bia[center_name]
                # ha = 'center'
            ax.text(
                label_x,
                i,
                f"{count}\n({percent}%)",
                va='center',
                ha='center',
                fontsize=10,
                color='black'
            )
            # print(percent)
            # print(label_x)
        # plt.subplots_adjust(top=0.95, bottom=0.1)

        plt.title(f"Missing Data Distribution in {center_name} Data", fontsize=16)
        plt.xlabel("Missing Percentage (%)", fontsize=12)
        plt.ylabel("Features", fontsize=12)

        max_percent = viz_df['Missing %'].max()
        plt.xlim(0, max_percent * 1.2)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # plt.tight_layout(pad=2.0)

        output_path = os.path.join(output_dir, f"Missing Data Distribution in {center_name} Data.tiff")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        # plt.show()
        plt.close()

# Missingness Heatmap
def missing_heatmap(center_name, center_data):
    # fig_height = max(8, min(20, len(center_data) * 0.03))
    fig = plt.figure(figsize=(16, 12))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.3, right=0.95)
    msno.matrix(center_data, sparkline=False, fontsize=12, color=(0.5294, 0.8078, 0.9216))  # purple: (0.52,0.43,1)
    plt.xticks(rotation=18, fontsize=11)
    plt.yticks([])
    plt.ylabel("Samples", fontsize=12, labelpad=15)
    plt.title(f"Missingness Heatmap in {center_name} Data", fontsize=20, pad=2)  # 12
    plt.savefig(f'Missingness Heatmap in {center_name} Data.tiff', dpi=300, format='tiff')
    # plt.show()
    plt.close()
