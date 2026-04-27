from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from rossmann_mlops.config import load_config, resolve_path


# ERRORS

class ProcessingError(ValueError):
    pass


class FeatureEngineeringError(ValueError):
    pass


# PATHS

DEFAULT_PATHS = {
    "store_raw":"data/raw/store.csv",
    "train_raw":"data/raw/train.csv",
    "test_raw":"data/raw/test.csv",

    "train_final":"data/processed/train_final.csv",
    "val_final":"data/processed/val_final.csv",
    "test_final":"data/processed/test_final.csv"
}


# PROCESSING

BASE_REQUIRED_COLUMNS = [
    "Store","DayOfWeek","Date",
    "Open","Promo",
    "StateHoliday","SchoolHoliday"
]

TRAIN_ONLY_REQUIRED_COLUMNS=["Sales"]

STORE_REQUIRED_COLUMNS=[
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "PromoInterval"
]


def _ensure_required_columns(
    frame,
    required_columns,
    source_name
):
    missing=[
        c for c in required_columns
        if c not in frame.columns
    ]

    if missing:
        raise ProcessingError(
            f"Missing columns in {source_name}: {missing}"
        )


def load_data(
    store_path,
    train_path,
    test_path
):

    store_path=resolve_path(store_path)
    train_path=resolve_path(train_path)
    test_path=resolve_path(test_path)

    store_df=pd.read_csv(store_path)

    train_df=pd.read_csv(
        train_path,
        dtype={"StateHoliday":str}
    )

    test_df=pd.read_csv(
        test_path,
        dtype={"StateHoliday":str}
    )

    _ensure_required_columns(
        store_df,
        STORE_REQUIRED_COLUMNS,
        "store"
    )

    _ensure_required_columns(
        train_df,
        BASE_REQUIRED_COLUMNS+
        TRAIN_ONLY_REQUIRED_COLUMNS,
        "train"
    )

    _ensure_required_columns(
        test_df,
        BASE_REQUIRED_COLUMNS,
        "test"
    )

    return (
        store_df,
        train_df,
        test_df
    )


def merge_data(
    train_df,
    test_df,
    store_df
):

    train_merged=pd.merge(
        train_df,
        store_df,
        on="Store",
        how="left"
    )

    test_merged=pd.merge(
        test_df,
        store_df,
        on="Store",
        how="left"
    )

    return (
        train_merged,
        test_merged
    )


def merge_store_data(df, store_df):

    if "Store" not in df.columns:
        raise ProcessingError("Input data must include 'Store' column")
    if "Store" not in store_df.columns:
        raise ProcessingError("Store data must include 'Store' column")

    return df.merge(
        store_df,
        on="Store",
        how="left"
    )


def handle_outliers(train_df):

    data=train_df.copy()

    data["Sales_log"]=np.log1p(
        data["Sales"].clip(lower=0)
    )

    return data


def _normalize_common_columns(frame):

    data=frame.copy()

    data["Date"]=pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data=data.dropna(
        subset=["Date"]
    )

    data["CompetitionDistance"]=(
        pd.to_numeric(
            data["CompetitionDistance"],
            errors="coerce"
        ).fillna(0)
    )

    fill_zero_cols=[
        "Promo2SinceWeek",
        "Promo2SinceYear",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear"
    ]

    for c in fill_zero_cols:

        data[c]=(
            pd.to_numeric(
                data[c],
                errors="coerce"
            ).fillna(0)
        )

    data["PromoInterval"]=(
        data["PromoInterval"]
        .fillna("None")
    )

    data["StateHoliday"]=(
        data["StateHoliday"]
        .astype(str)
    )

    return data


def preprocess_data(
    train_df,
    test_df
):

    cleaned_train=_normalize_common_columns(
        train_df
    )

    cleaned_test=_normalize_common_columns(
        test_df
    )

    cleaned_train=cleaned_train[
        (cleaned_train["Open"]!=0)
        &
        (cleaned_train["Sales"]>0)
    ].copy()

    cleaned_train=handle_outliers(
        cleaned_train
    )

    return (
        cleaned_train,
        cleaned_test
    )


# FEATURE ENGINEERING

@dataclass(frozen=True)
class FeatureSpec:
    required_columns:list[str]
    drop_columns:list[str]


FEATURE_SPEC=FeatureSpec(
required_columns=[
"Store","Date","Promo",
"StateHoliday","SchoolHoliday",
"StoreType","Assortment",
"CompetitionDistance","Promo2",
"Promo2SinceWeek","Promo2SinceYear",
"CompetitionOpenSinceMonth",
"CompetitionOpenSinceYear",
"PromoInterval"
],

drop_columns=[
"Date",
"CompetitionOpenSinceMonth",
"CompetitionOpenSinceYear",
"Promo2SinceWeek",
"Promo2SinceYear",
"PromoInterval",
"Customers",
"Open"
]
)

CATEGORICAL_COLUMNS=[
"StoreType",
"Assortment",
"StateHoliday",
"Promo",
"SchoolHoliday",
"Promo2",
"Is_Promo2_Month"
]

NUMERIC_COLUMNS=[
"Store",
"DayOfWeek",
"Month",
"Day",
"Year",
"WeekOfYear",
"CompetitionDistance",
"Promo2Open_Month",
"CompetitionOpen_Month"
]

STATE_HOLIDAY_MAP={
"0":0,"a":1,"b":2,"c":3
}

STORE_TYPE_MAP={
"a":0,"b":1,"c":2,"d":3
}

ASSORTMENT_MAP={
"a":0,"b":1,"c":2
}

MONTH_NAME_MAP={
1:"Jan",2:"Feb",3:"Mar",4:"Apr",
5:"May",6:"Jun",7:"Jul",8:"Aug",
9:"Sept",10:"Oct",11:"Nov",12:"Dec"
}


def _coerce_input_types(frame):

    data=frame.copy()

    data["Promo2"]=(
        pd.to_numeric(
            data["Promo2"],
            errors="coerce"
        ).fillna(0)
        .astype(int)
    )

    data["StoreType"]=(
        data["StoreType"]
        .astype(str)
        .str.lower()
    )

    data["Assortment"]=(
        data["Assortment"]
        .astype(str)
        .str.lower()
    )

    return data


def _add_time_features(frame):

    data=frame.copy()

    data["Year"]=data["Date"].dt.year
    data["Month"]=data["Date"].dt.month
    data["Day"]=data["Date"].dt.day

    data["WeekOfYear"]=(
        data["Date"]
        .dt.isocalendar()
        .week.astype(int)
    )

    data["DayOfWeek"]=(
        data["Date"].dt.weekday+1
    )

    return data


def _add_promo_competition_features(frame):

    data=frame.copy()

    sales_weeks=(
        data["Year"]*52
        +data["WeekOfYear"]
    )

    promo_weeks=(
        data["Promo2SinceYear"]*52
        +data["Promo2SinceWeek"]
    )

    data["Promo2Open_Month"]=(sales_weeks-promo_weeks)/4

    sales_months=(
        data["Year"]*12
        +data["Month"]
    )

    comp_months=(
        data["CompetitionOpenSinceYear"]*12
        +data["CompetitionOpenSinceMonth"]
    )

    data["CompetitionOpen_Month"]=(sales_months-comp_months)

    data.loc[
        (data["Promo2"]==0)|
        (data["Promo2SinceYear"]==0),
        "Promo2Open_Month"
    ]=0

    data.loc[
        data["CompetitionOpenSinceYear"]==0,
        "CompetitionOpen_Month"
    ]=0

    data["Promo2Open_Month"]=data["Promo2Open_Month"].clip(0,24)
    data["CompetitionOpen_Month"]=data["CompetitionOpen_Month"].clip(0,24)

    return data


def _add_promo_interval_feature(frame):

    data=frame.copy()

    data["month_tmp"]=(
        data["Month"].map(MONTH_NAME_MAP)
    )

    promo_interval=(
        data["PromoInterval"]
        .str.replace(" ","")
        .str.split(",")
    )

    mask=(
        (data["Promo2"]==1)
        &
        (data["PromoInterval"]!="None")
    )

    data["Is_Promo2_Month"]=0

    if mask.any():

        flags=[
            int(m in p)
            for m,p in zip(
                data.loc[mask,"month_tmp"],
                promo_interval.loc[mask]
            )
        ]

        data.loc[
            mask,
            "Is_Promo2_Month"
        ]=flags

    return data.drop(columns=["month_tmp"])


def _encode_categorical(frame):

    data=frame.copy()

    data["StateHoliday"]=(
        data["StateHoliday"]
        .map(STATE_HOLIDAY_MAP)
        .fillna(0)
        .astype(int)
    )

    data["StoreType"]=(
        data["StoreType"]
        .map(STORE_TYPE_MAP)
        .fillna(0)
        .astype(int)
    )

    data["Assortment"]=(
        data["Assortment"]
        .map(ASSORTMENT_MAP)
        .fillna(0)
        .astype(int)
    )

    return data


def build_features(df):

    data=_coerce_input_types(df)
    data=_add_time_features(data)
    data=_add_promo_competition_features(data)
    data=_add_promo_interval_feature(data)
    data=_encode_categorical(data)

    data=data.drop(
        columns=FEATURE_SPEC.drop_columns,
        errors="ignore"
    )

    return data


# SPLIT


def split_train_validation(train_df):

    val_condition=(
        (train_df["Year"]==2015)
        &
        (train_df["WeekOfYear"]>=26)
    )

    train_set=train_df[
        ~val_condition
    ].copy()

    val_set=train_df[
        val_condition
    ].copy()

    return train_set,val_set


# TARGET ENCODING + COLUMN ALIGNMENT

def add_target_encoding(
    train_set,
    val_set,
    test_df
):

    kf=KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_set["Store_DW_Promo_Avg"]=np.nan
    train_set["Month_Avg_Sales"]=np.nan

    for tr_idx,val_idx in kf.split(train_set):

        fold_train=train_set.iloc[tr_idx]
        fold_val=train_set.iloc[val_idx]

        store_avg=(
            fold_train.groupby(
                ["Store","DayOfWeek","Promo"]
            )["Sales_log"].mean()
        )

        train_set.loc[
            train_set.index[val_idx],
            "Store_DW_Promo_Avg"
        ]=(
            fold_val.set_index(
                ["Store","DayOfWeek","Promo"]
            ).index.map(store_avg)
        )

        month_avg=(
            fold_train.groupby(
                "Month"
            )["Sales_log"].mean()
        )

        train_set.loc[
            train_set.index[val_idx],
            "Month_Avg_Sales"
        ]=(
            fold_val["Month"]
            .map(month_avg)
        )


    store_avg=(
        train_set.groupby(
            ["Store","DayOfWeek","Promo"]
        )["Sales_log"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Sales_log":"Store_DW_Promo_Avg"
            }
        )
    )

    month_avg=(
        train_set.groupby(
            "Month"
        )["Sales_log"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "Sales_log":"Month_Avg_Sales"
            }
        )
    )


    val_set=val_set.merge(
        store_avg,
        on=["Store","DayOfWeek","Promo"],
        how="left"
    )

    val_set=val_set.merge(
        month_avg,
        on="Month",
        how="left"
    )

    test_df=test_df.merge(
        store_avg,
        on=["Store","DayOfWeek","Promo"],
        how="left"
    )

    test_df=test_df.merge(
        month_avg,
        on="Month",
        how="left"
    )


    global_mean=train_set["Sales_log"].mean()

    for df in [train_set,val_set,test_df]:

        df["Store_DW_Promo_Avg"]=(
            df["Store_DW_Promo_Avg"]
            .fillna(global_mean)
        )

        df["Month_Avg_Sales"]=(
            df["Month_Avg_Sales"]
            .fillna(global_mean)
        )


    for df in [train_set,val_set,test_df]:
        df.drop(
            columns=["Promo2"],
            errors="ignore",
            inplace=True
        )


    # FIX ĐỒNG BỘ CỘT
    feature_cols=[
        c for c in train_set.columns
        if c not in [
            "Sales",
            "Sales_log"
        ]
    ]

    val_set=val_set[
        feature_cols+["Sales_log"]
    ]

    test_df=test_df[
        feature_cols
    ]


    return (
        train_set,
        val_set,
        test_df
    )


# FULL PIPELINE

def run_pipeline(
    config_source="configs/config.yaml"
):

    load_config(config_source)

    # 1 processing
    store_df,train_df,test_df=load_data(
        DEFAULT_PATHS["store_raw"],
        DEFAULT_PATHS["train_raw"],
        DEFAULT_PATHS["test_raw"]
    )

    train_merged,test_merged=merge_data(
        train_df,
        test_df,
        store_df
    )

    train_processed,test_processed=preprocess_data(
        train_merged,
        test_merged
    )


    # 2 feature
    train_features=build_features(
        train_processed
    )

    test_features=build_features(
        test_processed
    )


    # 3 split
    train_set,val_set=split_train_validation(
        train_features
    )


    # 4 encoding
    train_set,val_set,test_df=add_target_encoding(
        train_set,
        val_set,
        test_features
    )


    # 5 export
    Path("data/processed").mkdir(
        parents=True,
        exist_ok=True
    )

    train_set.to_csv(
        DEFAULT_PATHS["train_final"],
        index=False
    )

    val_set.to_csv(
        DEFAULT_PATHS["val_final"],
        index=False
    )

    test_df.to_csv(
        DEFAULT_PATHS["test_final"],
        index=False
    )


    return {
        "train_rows":len(train_set),
        "val_rows":len(val_set),
        "test_rows":len(test_df)
    }


# MAIN

def main():

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/config.yaml"
    )

    args=parser.parse_args()

    result=run_pipeline(
        args.config
    )

    print(
        json.dumps(
            result,
            indent=2
        )
    )


if __name__=="__main__":
    main()