import pandas as pd

# Load the train data
try:
    train_df = pd.read_csv('train.csv')
    
    # Print information about the data
    print("Train data shape:", train_df.shape)
    print("\nFirst 5 rows:")
    print(train_df.head())
    
    print("\nColumn names:")
    print(train_df.columns.tolist())
    
    print("\nActivity distribution:")
    if 'activity' in train_df.columns:
        print(train_df['activity'].value_counts())
    
    print("\nData types:")
    print(train_df.dtypes)
    
except Exception as e:
    print(f"Error loading train.csv: {e}")
