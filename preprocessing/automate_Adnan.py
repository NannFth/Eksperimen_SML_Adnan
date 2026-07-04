import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path, output_dir):
    df = pd.read_csv(input_path)

    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())

    df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    X = df_encoded.drop('median_house_value', axis=1)
    y = df_encoded['median_house_value']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    os.makedirs(output_dir, exist_ok=True)

    train_data = X_train_scaled.copy()
    train_data['median_house_value'] = y_train.values
    train_data.to_csv(os.path.join(output_dir, 'train.csv'), index=False)

    test_data = X_test_scaled.copy()
    test_data['median_house_value'] = y_test.values
    test_data.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print("Preprocessing selesai!")
    print("Train:", train_data.shape)
    print("Test:", test_data.shape)

    return train_data, test_data


if __name__ == "__main__":
    input_path = os.path.join("..", "california_housing_raw", "housing.csv")
    output_dir = "california_housing_preprocessing"

    preprocess_data(input_path, output_dir)