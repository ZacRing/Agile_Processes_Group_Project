import kagglehub as kh
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# boxplots to prove the outliers are there, any dots outside the whiskers are outliers
# each column gets its own plot cos theyre all on different scales
outlier_cols = ["height", "weight", "ap_hi", "ap_lo"]


def plot_boxplots(data, title, filename):
    fig, axes = plt.subplots(1, len(outlier_cols), figsize=(16, 5))
    for ax, col in zip(axes, outlier_cols):
        sns.boxplot(y=data[col], ax=ax, color="skyblue")
        ax.set_title(col)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.show()


plot_boxplots(df, "Before removing outliers", "boxplots_before_cleaning.png")

rows_before = len(df)

# anything outside these is a typo nobody is 55cm tall or has a bp of 160/20
# height and weight can each look fine but be impossible together, bmi catches that
bmi = df["weight"] / (df["height"] / 100) ** 2
df = df[(bmi >= 13) & (bmi <= 60)]
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 160)]

# top bp number has to be bigger than the bottom one if not its typed wrong
df = df[df["ap_hi"] > df["ap_lo"]]

print("Outlier rows removed:", rows_before - len(df))

# same boxplots again after so you can see the crazy values (like bp of 11000 or height of 55cm) are gone
plot_boxplots(df, "After removing outliers", "boxplots_after_cleaning.png")

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


# correlation between columns
print("\ncorrelation")

# correlation goes from -1 to 1, close to 1 means two columns go up together, close to -1 means one goes up the other goes down, 0 means no link
corr = df.corr()

# how much each feature is linked to cardio (what we are predicting)
print("Correlation with cardio:")
print(corr["cardio"].drop("cardio").sort_values(ascending=False).round(3))

# heatmap of every column against every other column
plt.figure(figsize=(11, 9))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=120)
plt.show()

# flag any pair of features that are very strongly linked as they basically give the model the same info twice
strong_pairs = []
feature_cols = [c for c in corr.columns if c != "cardio"]
for i, a in enumerate(feature_cols):
    for b in feature_cols[i + 1:]:
        if abs(corr.loc[a, b]) > 0.7:
            strong_pairs.append((a, b, round(corr.loc[a, b], 3)))
print("Feature pairs with correlation over 0.7:", strong_pairs if strong_pairs else "none")


# save
print("\nRows at the end:", len(df))
print(df.describe())
df.to_csv("heart_clean.csv", index=False)
print("Saved heart_clean.csv")