import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# Config
# =========================
INPUT_CSV = "../data/labels_mapped.csv"
TRAIN_CSV = "../data/labels_train.csv"
TEST_CSV = "../data/labels_test.csv"

TEST_SIZE = 0.1
RANDOM_STATE = 42


# =========================
# Load dataset
# =========================
df = pd.read_csv(INPUT_CSV, encoding="utf-8")

print(f"\n📊 Original dataset size: {len(df)}")

# -------------------------
# Sanity check
# -------------------------
if "main_genre" not in df.columns:
    raise ValueError("❌ 'main_genre' column not found. Mapping step missing.")

# Drop rows without main_genre (very important)
df = df[df["main_genre"].notna()]

print("\n📊 MAIN genre distribution (before split):")
print(df["main_genre"].value_counts())


# =========================
# Stratified split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["main_genre"]
)

# =========================
# Save CSVs
# =========================
train_df.to_csv(TRAIN_CSV, index=False, encoding="utf-8")
test_df.to_csv(TEST_CSV, index=False, encoding="utf-8")

print("\n✅ Stratified split completed!")
print(f"🟢 Train size: {len(train_df)}")
print(f"🔵 Test size: {len(test_df)}")

print("\n📊 Train genre distribution:")
print(train_df["main_genre"].value_counts())

print("\n📊 Test genre distribution:")
print(test_df["main_genre"].value_counts())
