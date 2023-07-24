# Read the CSV file
import pandas as pd

# Read the CSV file
df_full = pd.read_csv('experiment_report_low_quality_images.csv')

# Calculate the Metric** for "No preprocessing" for each image
df_no_preprocessing_wer = df_full[df_full['Preprocessing Steps'] == 'No preprocessing']
no_preprocessing_wer = df_no_preprocessing_wer.set_index('Filename')['WER']

df_no_preprocessing_cer = df_full[df_full['Preprocessing Steps'] == 'No preprocessing']
no_preprocessing_cer = df_no_preprocessing_cer.set_index('Filename')['CER']

df_no_preprocessing_levenshtein = df_full[df_full['Preprocessing Steps'] == 'No preprocessing']
no_preprocessing_levenshtein = df_no_preprocessing_levenshtein.set_index('Filename')['Levenshtein Distance']

# Join this with the original dataframe
df_full_wer = df_full.set_index('Filename').join(no_preprocessing_wer, rsuffix='_no_preprocessing')
df_full_cer = df_full.set_index('Filename').join(no_preprocessing_cer, rsuffix='_no_preprocessing')
df_full_levenshtein = df_full.set_index('Filename').join(no_preprocessing_levenshtein, rsuffix='_no_preprocessing')


# Calculate the improvement in WER compared to "No preprocessing"
df_full_wer['Improvement in WER Percent'] = (df_full_wer['WER_no_preprocessing'] - df_full_wer['WER']) / df_full_wer['WER_no_preprocessing'] * 100
df_full_cer['Improvement in CER Percent'] = (df_full_cer['CER_no_preprocessing'] - df_full_cer['CER']) / df_full_cer['CER_no_preprocessing'] * 100
df_full_levenshtein['Improvement in Levenshtein Distance Percent'] = (df_full_levenshtein['Levenshtein Distance_no_preprocessing'] - df_full_levenshtein['Levenshtein Distance']) / df_full_levenshtein['Levenshtein Distance_no_preprocessing'] * 100


# Calculate the average and median improvement for each preprocessing step
average_improvement_wer = df_full_wer.groupby('Preprocessing Steps')['Improvement in WER Percent'].mean()
median_improvement_wer = df_full_wer.groupby('Preprocessing Steps')['Improvement in WER Percent'].median()
best_improvement_wer = df_full_wer.groupby('Preprocessing Steps')['Improvement in WER Percent'].max()
worst_improvement_wer = df_full_wer.groupby('Preprocessing Steps')['Improvement in WER Percent'].min()

average_improvement_cer = df_full_cer.groupby('Preprocessing Steps')['Improvement in CER Percent'].mean()
median_improvement_cer = df_full_cer.groupby('Preprocessing Steps')['Improvement in CER Percent'].median()
best_improvement_cer = df_full_cer.groupby('Preprocessing Steps')['Improvement in CER Percent'].max()
worst_improvement_cer = df_full_cer.groupby('Preprocessing Steps')['Improvement in CER Percent'].min()

average_improvement_levenshtein = df_full_levenshtein.groupby('Preprocessing Steps')['Improvement in Levenshtein Distance Percent'].mean()
median_improvement_levenshtein = df_full_levenshtein.groupby('Preprocessing Steps')['Improvement in Levenshtein Distance Percent'].median()
best_improvement_levenshtein = df_full_levenshtein.groupby('Preprocessing Steps')['Improvement in Levenshtein Distance Percent'].max()
worst_improvement_levenshtein = df_full_levenshtein.groupby('Preprocessing Steps')['Improvement in Levenshtein Distance Percent'].min()


# Combine average and median improvements into a single DataFrame
improvements_wer = pd.DataFrame({
    'Average Improvement (%)': average_improvement_wer,
    'Median Improvement (%)': median_improvement_wer,
    'Best Improvement (%)': best_improvement_wer,
    'Worst Improvement (%)': worst_improvement_wer
})

improvements_cer = pd.DataFrame({
    'Average Improvement (%)': average_improvement_cer,
    'Median Improvement (%)': median_improvement_cer,
    'Best Improvement (%)': best_improvement_cer,
    'Worst Improvement (%)': worst_improvement_cer
})

improvements_levenshtein = pd.DataFrame({
    'Average Improvement (%)': average_improvement_levenshtein,
    'Median Improvement (%)': median_improvement_levenshtein,
    'Best Improvement (%)': best_improvement_levenshtein,
    'Worst Improvement (%)': worst_improvement_levenshtein
})

# Save the DataFrame to a CSV file
improvements_wer.to_csv('improvements_wer_all_combinations.csv')
improvements_cer.to_csv('improvements_cer_all_combinations.csv')
improvements_levenshtein.to_csv('improvements_levenshtein-distance_all_combinations.csv')
