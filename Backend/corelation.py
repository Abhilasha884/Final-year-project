import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/labels_mapped.csv")

# Encode main_genre numerically
genre_codes = {g: i for i, g in enumerate(df["main_genre"].unique())}
df["main_genre_code"] = df["main_genre"].map(genre_codes)

# Correlation matrix
corr = df[["valence", "arousal", "main_genre_code"]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Genre and Emotion Dimensions")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(
    x="main_genre",
    y="valence",
    data=df,
    estimator="mean",
    ci=None
)

plt.xticks(rotation=45)
plt.ylabel("Mean Valence")
plt.title("Average Valence Across Music Genres")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    x="valence",
    y="arousal",
    hue="main_genre",
    data=df,
    alpha=0.7
)

plt.title("Valence–Arousal Distribution by Genre")
plt.xlabel("Valence (Positive ↔ Negative)")
plt.ylabel("Arousal (Calm ↔ Energetic)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

