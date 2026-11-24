"""Dataset helpers for NSL-KDD: column names, attack grouping and file loading."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

TRAIN_FILE = DATA_DIR / "KDDTrain+.txt"
TEST_FILE = DATA_DIR / "KDDTest+.txt"

# The 41 features of one network connection (+ label and difficulty level).
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
COLUMNS = FEATURES + ["label", "difficulty"]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Every specific attack name belongs to one of 4 attack families.
ATTACK_FAMILIES = {
    "DoS": [  # Denial of Service: flood a machine so it stops responding
        "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
        "udpstorm", "processtable", "worm", "mailbomb"],
    "Probe": [  # Scanning a network to find weaknesses
        "satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"],
    "R2L": [  # Remote to Local: outsider tries to get local access
        "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
        "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
        "httptunnel", "sendmail", "named"],
    "U2R": [  # User to Root: normal user tries to become admin
        "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
        "xterm", "ps"],
}
_LABEL_TO_CATEGORY = {"normal": "Normal"}
for family, names in ATTACK_FAMILIES.items():
    for name in names:
        _LABEL_TO_CATEGORY[name] = family

CATEGORY_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]


def to_category(label):
    """Map a raw label such as 'neptune' to 'DoS' (or 'normal' to 'Normal')."""
    return _LABEL_TO_CATEGORY.get(str(label).strip().lower(), "Unknown")


def data_available():
    return TRAIN_FILE.exists() and TEST_FILE.exists()


def load_nsl_kdd(path):
    """Load an NSL-KDD file and add 'category' and 'is_attack' columns."""
    df = pd.read_csv(path, names=COLUMNS)
    df["category"] = df["label"].map(to_category)
    df["is_attack"] = (df["category"] != "Normal").astype(int)
    return df


def read_traffic_file(path):
    """Read a user's CSV of connections (with or without a header row)."""
    df = pd.read_csv(path)
    if not set(FEATURES).issubset(df.columns):
        df = pd.read_csv(path, header=None)
        if df.shape[1] in (41, 42, 43):
            df.columns = COLUMNS[: df.shape[1]]
        else:
            raise ValueError(
                "File must have the 41 NSL-KDD feature columns "
                f"(found {df.shape[1]} columns)."
            )
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing[:5])} ...")
    return df
