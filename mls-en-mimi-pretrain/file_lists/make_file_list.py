# make_file_list.py
from huggingface_hub import list_repo_files

REPO_ID = "parler-tts/mls_eng"
files = [f for f in list_repo_files(REPO_ID, repo_type="dataset") if f.endswith(".parquet")]
files = sorted(files)

train_files = [f for f in files if "train" in f]
dev_files = [f for f in files if "dev" in f]
test_files = [f for f in files if "test" in f]

with open("train_files.txt", "w") as f:
    f.write("\n".join(train_files))
print(f"Wrote {len(train_files)} paths to train_files.txt")

with open("dev_files.txt", "w") as f:
    f.write("\n".join(dev_files))
print(f"Wrote {len(dev_files)} paths to dev_files.txt")

with open("test_files.txt", "w") as f:
    f.write("\n".join(test_files))
print(f"Wrote {len(test_files)} paths to test_files.txt")