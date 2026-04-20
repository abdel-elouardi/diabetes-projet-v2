import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def get_info(df):
    print("=== .info() ===")
    print(df.info())
    print("\n=== .describe() ===")
    print(df.describe())
    print("\n=== .dtypes ===")
    print(df.dtypes)
    print("\n=== Valeurs manquantes ===")
    print(df.isna().sum())
    return df

def clean_columns(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(' ', '_')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(',', '.', regex=False)
    print("✅ Colonnes nettoyées")
    return df

def fix_types(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        except:
            pass
    print("✅ Types vérifiés")
    return df

def check_duplicates(df):
    duplicates = df.duplicated().sum()
    print(f"\n=== Doublons ===")
    print(f"Nombre de doublons : {duplicates}")
    if duplicates > 0:
        df = df.drop_duplicates()
        print("✅ Doublons supprimés !")
    else:
        print("✅ Aucun doublon trouvé")
    return df

def handle_missing(df):
    print("\n=== Valeurs manquantes ===")
    print(df.isna().sum())
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            print(f"✅ {col} → remplacé par la médiane")
    return df

def fix_impossible_values(df):
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    existing_cols = [col for col in cols_with_zeros if col in df.columns]
    print("\n=== Valeurs impossibles (zéros) ===")
    for col in existing_cols:
        zeros = (df[col] == 0).sum()
        if zeros > 0:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())
            print(f"✅ {col} : {zeros} zéros remplacés par la médiane")
    return df

def remove_outliers(df):
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Outcome' in cols:
        cols.remove('Outcome')
    print("\n=== Outliers (méthode IQR) ===")
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
        df[col] = df[col].clip(lower, upper)
        print(f"✅ {col} : {outliers} outliers clippés")
    return df

def clean_data(df):
    print("🧹 Début du nettoyage...\n")
    df = clean_columns(df)
    df = fix_types(df)
    df = check_duplicates(df)
    df = fix_impossible_values(df)
    df = handle_missing(df)
    df = remove_outliers(df)
    print("\n✅ Nettoyage terminé !")
    return df