from pathlib import Path
import pandas as pd
import os


df1 = pd.read_csv("cleaned_datasets/dataset_1_cleaned.csv", sep=";", low_memory=False)
df2 = pd.read_csv("cleaned_datasets/dataset_2_cleaned.csv", sep=";", low_memory=False)
df3 = pd.read_csv("cleaned_datasets/dataset_3_cleaned.csv", sep=";", low_memory=False)

os.makedirs("new_datasets/merged_raw", exist_ok=True)

#Create train dataset concatenating dataset 1 and legitimate samples of dataset 2
# legitimate samples of dataset 2 are those with label ip.opt.time_stamp = NaN,
df_train = pd.concat([df1, df2[df2["ip.opt.time_stamp"].isna()]], ignore_index=True)
# printing the shape of the train dataset
print("Train dataset shape:", df_train.shape)
# saving the train dataset to csv
df_train.to_csv("new_datasets/merged_raw/train_dataset.csv", sep=";", index=False)

# Create test dataset with all samples of dataset 3 and attack samples of dataset 2
df_test = pd.concat([df2[~df2["ip.opt.time_stamp"].isna()], df3], ignore_index=True)
# printing the shape of the test dataset
print("Test dataset shape:", df_test.shape)
# saving the test dataset to csv
df_test.to_csv("new_datasets/merged_raw/test_dataset.csv", sep=";", index=False)