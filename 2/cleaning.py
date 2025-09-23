import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("HousingWhichNeedsCleaning.csv")
print("Initial rows:", len(df))

data = df[['area', 'price']].copy()

data['area'] = data['area'].astype(str).str.replace(r'[^0-9.-]', '', regex=True)

data['area'] = pd.to_numeric(data['area'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')
print("After cleaning text & converting to numeric:", len(data))

data['area'].fillna(data['area'].mean(), inplace=True)
data['price'].fillna(data['price'].mean(), inplace=True)
print("After filling missing values:", len(data))

data.drop_duplicates(inplace=True)
print("After removing duplicates:", len(data))

data = data[(data['area'] >= 0) & (data['price'] >= 0)]
print("After removing negatives:", len(data))

for col in ['area', 'price']:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(data)
    data = data[(data[col] >= lower) & (data[col] <= upper)]
    print(f"After outlier removal on {col}:", len(data), f"(removed {before - len(data)})")

scaler = MinMaxScaler()
data[['area', 'price']] = scaler.fit_transform(data[['area', 'price']])
print("After normalization:", len(data))

print("\n✅ Final Cleaned Data Statistics:")
print(data.describe())

data.to_csv("Cleaned_Housing_Data.csv", index=False)
print("\n💾 Cleaned dataset saved as 'Cleaned_Housing_Data.csv'")
