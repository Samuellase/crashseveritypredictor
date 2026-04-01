import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
import os
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)


def load_data(data_path: str, nrows: int = 500000) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(data_path, nrows=nrows)
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    print(f"Dataset shape: {df.shape}")
    print("\nSeverity distribution:")
    print(df['Severity'].value_counts().sort_index())
    return df


def check_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    print("\nTop 10 columns with missing values:")
    print(missing[missing > 0].sort_values(ascending=False).head(10))


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    # Binary target: Low (1-2) vs High (3-4) severity
    # Following Delen et al. 2017 methodology
    df['Severity_Binary'] = df['Severity'].apply(lambda x: 0 if x <= 2 else 1)
    print("\nBinary severity created:")
    print(df['Severity_Binary'].value_counts())
    print("\nPercentage:")
    print(df['Severity_Binary'].value_counts(normalize=True).round(3))
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        'Severity_Binary',
        # Weather
        'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
        'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)',
        'Weather_Condition',
        # Time
        'Start_Time',
        # Road features
        'Junction', 'Traffic_Signal', 'Stop', 'Crossing',
        # Light conditions
        'Sunrise_Sunset', 'Civil_Twilight'
    ]
    available = [f for f in features if f in df.columns]
    df = df[available].copy()
    print(f"\nSelected {len(available)} features | Shape: {df.shape}")
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['Severity_Binary'])

    numeric_cols = ['Temperature(F)', 'Humidity(%)', 'Pressure(in)',
                    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']
    for col in numeric_cols:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    cat_cols = ['Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight',
                'Junction', 'Traffic_Signal', 'Stop', 'Crossing']
    for col in cat_cols:
        if col in df.columns:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

    print(f"Missing values after imputation: {df.isnull().sum().sum()}")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'Sunrise_Sunset' in df.columns:
        df['Sunrise_Sunset'] = df['Sunrise_Sunset'].map({'Day': 1, 'Night': 0}).fillna(0)

    if 'Civil_Twilight' in df.columns:
        df['Civil_Twilight'] = df['Civil_Twilight'].map({'Day': 1, 'Night': 0}).fillna(0)

    bool_cols = ['Junction', 'Traffic_Signal', 'Stop', 'Crossing']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(0)

    if 'Weather_Condition' in df.columns:
        top_weather = df['Weather_Condition'].value_counts().head(10).index
        df['Weather_Condition'] = df['Weather_Condition'].apply(
            lambda x: x if x in top_weather else 'Other'
        )
        df = pd.get_dummies(df, columns=['Weather_Condition'], prefix='Weather', drop_first=True)

    print(f"After encoding: {df.shape}")
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'Start_Time' not in df.columns:
        return df

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    def categorize_hour(hour):
        if pd.isna(hour):
            return 0
        elif 6 <= hour < 12:
            return 1   # Morning
        elif 12 <= hour < 18:
            return 2   # Afternoon
        elif 18 <= hour < 22:
            return 3   # Evening
        else:
            return 4   # Night

    df['Hour_Category'] = df['Hour'].apply(categorize_hour)
    df = df.drop('Start_Time', axis=1)
    print("Time features created: Hour, DayOfWeek, Month, IsWeekend, Hour_Category")
    return df


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    low_severity = df[df['Severity_Binary'] == 0]
    high_severity = df[df['Severity_Binary'] == 1]

    print(f"\nBefore balancing:")
    print(f"  Low severity:  {len(low_severity):,}")
    print(f"  High severity: {len(high_severity):,}")

    if len(low_severity) > len(high_severity):
        low_sampled = resample(low_severity, n_samples=len(high_severity),
                               random_state=42, replace=False)
        df_balanced = pd.concat([low_sampled, high_severity])
    else:
        high_sampled = resample(high_severity, n_samples=len(low_severity),
                                random_state=42, replace=False)
        df_balanced = pd.concat([low_severity, high_sampled])

    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nAfter balancing:")
    print(df_balanced['Severity_Binary'].value_counts())
    return df_balanced


def plot_severity_distribution(df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(8, 5))
    df['Severity_Binary'].value_counts().plot(kind='bar', color=['skyblue', 'coral'])
    plt.title('Crash Severity Distribution')
    plt.xlabel('Severity (0=Low, 1=High)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    raw_path = os.path.join('data', 'raw', 'US_Accidents_March23.csv')
    processed_path = os.path.join('data', 'processed', 'clean_crash_data.csv')
    results_path = os.path.join('results', 'severity_distribution.png')

    if not os.path.exists(raw_path):
        print("ERROR: Dataset not found!")
        print(f"Expected location: {raw_path}")
        print("\nDownload from: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")
        raise FileNotFoundError("Dataset missing")

    df = load_data(raw_path)
    check_missing(df)
    df = create_binary_target(df)
    plot_severity_distribution(df, results_path)
    df = select_features(df)
    df = impute_missing(df)
    df = encode_features(df)
    df = extract_time_features(df)
    df = df.dropna()
    df = balance_classes(df)

    print(f"\nFINAL DATASET SUMMARY")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target distribution:\n{df['Severity_Binary'].value_counts()}")

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_csv(processed_path, index=False)
    print(f"\nPreprocessing complete. Saved: {processed_path}")


if __name__ == '__main__':
    main()
