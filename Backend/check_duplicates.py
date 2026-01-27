import pandas as pd

TRAIN_CSV = "data/labels_train.csv"
TEST_CSV = "data/labels_test.csv"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_ids = set(train_df["song_id"])
test_ids = set(test_df["song_id"])

duplicates = train_ids.intersection(test_ids)

print(f"Train size: {len(train_ids)}")
print(f"Test size: {len(test_ids)}")
print(f"Duplicates found: {len(duplicates)}")

if duplicates:
    print("❌ DUPLICATE SONG IDs FOUND:")
    print(list(duplicates)[:10])
else:
    print("✅ NO DUPLICATES — Train & Test are clean")


# Count per genre
train_counts = train_df['main_genre'].value_counts()
test_counts  = test_df['main_genre'].value_counts()
total_counts = train_counts + test_counts

# Compute percentages
train_pct = (train_counts / total_counts) * 100
test_pct  = (test_counts / total_counts) * 100

result = pd.DataFrame({
    'Train_Count': train_counts,
    'Test_Count': test_counts,
    'Train_%': train_pct.round(2),
    'Test_%': test_pct.round(2)
})

print(result)
