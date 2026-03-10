import pandas as pd

try:
    df4 = pd.read_csv('results/stage4a_rho_values.csv')
    df5 = pd.read_csv('results/stage5_confusion_delta.csv')
    
    summary = pd.merge(df4, df5, on='k', how='left')
    summary.to_csv('results/summary.csv', index=False)
    print("Created summary.csv")
except Exception as e:
    print(f"Error: {e}")
