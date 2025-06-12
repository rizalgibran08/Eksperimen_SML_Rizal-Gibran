import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


def preprocess_data(df):
    # Drop kolom tidak relevan
    drop_cols = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Reason', 'Churn Label'
    ]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    # Ganti kolom target
    df['Churn'] = df['Churn Value']
    df.drop(columns=['Churn Value'], inplace=True)

    # Ubah TotalCharges ke float
    df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
    df['Total Charges'].fillna(df['Total Charges'].median(), inplace=True)

    # Drop CLTV karena sparsity
    if 'CLTV' in df.columns:
        df.drop(columns=['CLTV'], inplace=True)

    # Label Encoding untuk biner kategori
    # Encoding fitur biner
    binary_cols = [col for col in df.select_dtypes(
        include='object') if df[col].nunique() == 2]
    for col in binary_cols:
        mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
        df[col] = df[col].map(mapping).astype(int)

    # Encoding fitur multikategori
    multi_cols = [col for col in df.select_dtypes(
        include='object') if df[col].nunique() > 2]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    # Pastikan tidak ada tipe bool
    df = df.astype(
        {col: 'int' for col in df.columns if df[col].dtype == 'bool'})

    # Feature scaling
    num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df


def save_preprocessed_data(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"[INFO] Dataset berhasil disimpan di {output_path}")


if __name__ == '__main__':
    input_file = r'Telco_customer_churn_raw.xlsx'
    output_file = 'Telco_preprocessed.csv'

    print("[INFO] Memuat data...")
    df_raw = load_data(input_file)

    print("[INFO] Melakukan preprocessing...")
    df_clean = preprocess_data(df_raw)

    print("[INFO] Menyimpan hasil...")
    save_preprocessed_data(df_clean, output_file)
