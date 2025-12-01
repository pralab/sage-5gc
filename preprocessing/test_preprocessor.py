from preprocessor import Preprocessor
from pathlib import Path
import pandas as pd

# ===============================================================
#                 🔍 1. CHECK INITIAL RAW DATASETS
# ===============================================================

train_raw_path = f"{Path(__file__).parent.parent}/data/cleaned_datasets/dataset_training_benign.csv"
test_raw_path  = f"{Path(__file__).parent.parent}/data/cleaned_datasets/dataset_3_cleaned.csv"

df_raw_train = pd.read_csv(train_raw_path, sep=";", low_memory=False)
df_raw_test  = pd.read_csv(test_raw_path,  sep=";", low_memory=False)

print("\n=================== RAW DATASET COLUMN CHECK ===================\n")

raw_train_cols = set(df_raw_train.columns)
raw_test_cols  = set(df_raw_test.columns)

if raw_train_cols == raw_test_cols:
    print("✅ RAW Train and Test datasets have EXACTLY the same columns.\n")
    print("Columns Number:", len(raw_train_cols))
else:
    print("❌ RAW COLUMN MISMATCH detected!\n")
    print("Columns Number RAW TRAIN:", len(raw_train_cols))
    print("Columns Number RAW TEST :", len(raw_test_cols))

    missing_in_train_raw = raw_test_cols - raw_train_cols
    missing_in_test_raw  = raw_train_cols - raw_test_cols

    if missing_in_train_raw:
        print("⚠️ Columns present in RAW TEST but missing in RAW TRAIN:")
        for c in sorted(missing_in_train_raw):
            print(f"   + {c}")

    if missing_in_test_raw:
        print("\n⚠️ Columns present in RAW TRAIN but missing in RAW TEST:")
        for c in sorted(missing_in_test_raw):
            print(f"   - {c}")

# ===============================================================
#                🔧 2. RUN PREPROCESSING PIPELINE
# ===============================================================

pre = Preprocessor()

print("\n=================== RUNNING TRAIN PIPELINE ===================\n")
df_train = pre.train(
    output_dir="processed_data_train",
    input_file=train_raw_path,
)

print("\n==================== RUNNING TEST PIPELINE ====================\n")
df_test = pre.test(
    output_dir="processed_data_test",
    input_file=test_raw_path,
)

# ===============================================================
#                🔍 3. CHECK FINAL PROCESSED COLUMNS
# ===============================================================

print("\n=================== FINAL COLUMN CHECK ===================\n")

train_cols = set(df_train.columns)
test_cols  = set(df_test.columns)

if train_cols == test_cols:
    print("✅ FINAL Train and Test datasets have EXACTLY the same processed columns.\n")
    print("Columns Number:", len(train_cols))
else:
    print("❌ FINAL COLUMN MISMATCH detected!\n")
    print("Columns Number FINAL TRAIN:", len(train_cols))
    print("Columns Number FINAL TEST :", len(test_cols))

    missing_in_train = test_cols - train_cols
    missing_in_test  = train_cols - test_cols

    if missing_in_train:
        print("⚠️ Columns present in FINAL TEST but missing in FINAL TRAIN:")
        for c in sorted(missing_in_train):
            print(f"   + {c}")

    if missing_in_test:
        print("\n⚠️ Columns present in FINAL TRAIN but missing in FINAL TEST:")
        for c in sorted(missing_in_test):
            print(f"   - {c}")
