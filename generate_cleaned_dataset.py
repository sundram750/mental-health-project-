import pandas as pd
import numpy as np

np.random.seed(42)
N = 1500

df = pd.DataFrame({
    'hr': np.random.normal(75, 10, N).clip(50, 120).round(1),
    'hrv': np.random.normal(65, 20, N).clip(15, 200).round(1),
    'skin_temp': np.random.normal(35.2, 0.8, N).clip(32.0, 38.0).round(2),
    'respiration_rate': np.random.normal(16, 4, N).clip(8, 40).round(1),
    'bp_systolic': np.random.normal(118, 15, N).clip(80, 200).astype(int),
    'bp_diastolic': np.random.normal(76, 10, N).clip(40, 120).astype(int),
    'cognitive_state': np.random.randint(1, 6, N),
    'emotional_state': np.random.randint(1, 6, N)
})

file_name = 'cleaned_dataset_1500.csv'
df.to_csv(file_name, index=False)
print(f"Created {file_name} with {len(df)} rows")
