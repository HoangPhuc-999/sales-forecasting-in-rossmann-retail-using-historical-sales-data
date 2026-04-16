import pandas as pd
import numpy as np

# ===== LOAD DATA =====
def load_data(store_path, train_path, test_path):
    store = pd.read_csv(store_path)
    train = pd.read_csv(train_path, dtype={"StateHoliday": str})
    test = pd.read_csv(test_path, dtype={"StateHoliday": str})
    return store, train, test

# ===== MERGE DATA =====
def merge_data(train, test, store):
    return (
        pd.merge(train, store, on='Store', how='left'),
        pd.merge(test, store, on='Store', how='left')
    )
# ===== OUTLIER =====
def handle_outliers(df):
    df['Sales_log'] = np.log1p(df['Sales'])
    return df
# ===== PREPROCESS =====
def preprocess_data(train, test):
    for df in [train, test]:
        df['Date'] = pd.to_datetime(df['Date'])
        df['CompetitionDistance'] = df['CompetitionDistance'].fillna(0)
    train['StateHoliday'] = train['StateHoliday'].map(
        lambda x: str(x) if pd.notnull(x) else x
    )

    # log feature
    train['CompetitionDistance_log'] = np.log1p(train['CompetitionDistance'])
    test['CompetitionDistance_log'] = np.log1p(test['CompetitionDistance'])

    cols_fill_zero = [
        'Promo2SinceWeek', 'Promo2SinceYear',
        'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'
    ]

    for df in [train, test]:
        df[cols_fill_zero] = df[cols_fill_zero].fillna(0)
        df['PromoInterval'] = df['PromoInterval'].fillna('None')

    # filter + target log
    train = train[(train['Open'] != 0) & (train['Sales'] > 0)]
    train = handle_outliers(train)

    return train, test


# ===== MAIN =====
def main():
    store, train, test = load_data("store.csv", "train.csv", "test.csv")

    train_m, test_m = merge_data(train, test, store)
    train_p, test_p = preprocess_data(train_m, test_m)

    print("Train processed:", train_p.shape)
    print("Test processed:", test_p.shape)

    # 🔥 SAVE FILE
    train_p.to_csv("train_processed.csv", index=False)
    test_p.to_csv("test_processed.csv", index=False)

    print("Saved processed data!")


if __name__ == "__main__":
    main()