"""Download the NSL-KDD dataset (train + test) into the data/ folder."""
import urllib.request

from nids.data_utils import DATA_DIR

BASE = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"KDDTrain%2B.txt": "KDDTrain+.txt", "KDDTest%2B.txt": "KDDTest+.txt"}

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for remote, local in FILES.items():
        target = DATA_DIR / local
        if target.exists():
            print(f"{local} already exists - skipping")
            continue
        print(f"Downloading {local} ...")
        urllib.request.urlretrieve(BASE + remote, target)
    print("Done. Files are in:", DATA_DIR)
