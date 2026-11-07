import kagglehub as kh
import pandas as pd

# Download latest version
path = kh.dataset_download("thedevastator/exploring-risk-factors-for-cardiovascular-diseas")
print("Path to dataset files:", path)

# load the data
df = pd.read_csv(path + "/heart_data.csv")
print("Rows at the start:", len(df))

# index and id are just row numbers so get rid of them but have to do this first or no rows would ever look like duplicates
df = df.drop(columns=["index", "id"])

# fill null values
print("\nfill null values")

# check for nulls
print(df.isna().sum())

# fill number columns with the median cos data isint cleaned yet so mean could be skewed heavily with drastic outliers like we havent checked height yet for example so someone could be 200000000cm tall which would mess it up completley
number_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
for col in number_cols:
    df[col] = df[col].fillna(df[col].median())

# delete rows missing a category value cant guess those
category_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
df = df.dropna(subset=category_cols)

# make sure theres no nulls left
print("Nulls left:", df.isna().sum().sum())


#  duplicates
print("\nduplicates")
print("Duplicates found:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates left:", df.duplicated().sum())


# catching outliers
print("\noutliers")

# check min and max before cleaning so we can see whats unrealistic
print("Min and max before removing outliers:")
print(df[["age", "height", "weight", "ap_hi", "ap_lo"]].agg(["min", "max"]))

rows_before = len(df)

# anything outside these is a typo nobody is 55cm tall or has a bp of 160/20
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]

# top bp number has to be bigger than the bottom one if not its typed wrong
df = df[df["ap_hi"] > df["ap_lo"]]

print("Outlier rows removed:", rows_before - len(df))

# age is in days for some reason so im changing it to years its more natural and add a .25 for leap years and i round it
df["age"] = (df["age"] / 365.25).round(1)
df = df.rename(columns={"age": "age_years"})

# check min and max before cleaning so we can see whats unrealistic
print("Min and max after removing outliers:")
print(df[["age_years", "height", "weight", "ap_hi", "ap_lo"]].agg(["min", "max"]))


#encode categorical values
print("\nencode categorical values")

# step 1 list text columns there arent any in this dataset
text_cols = df.select_dtypes(include="object").columns
print("Text columns:", list(text_cols))

# step 2 binary columns to 0 and 1 and gender is 1 for female 2 for male gonna swap them to 1 and 0 to keep it consistent
df["gender"] = df["gender"].replace({1: 0, 2: 1})
df = df.rename(columns={"gender": "is_male"})

#theres no text colums in this file so i dont need to hot encode anything even tho the task description says too

#make sure everything is a number now
print(df.dtypes)


# save
print("\nRows at the end:", len(df))
print(df.describe())
df.to_csv("heart_clean.csv", index=False)
print("Saved heart_clean.csv")