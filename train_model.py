"""Command-line training:  python train_model.py [--mode binary|multi|both] [--compare]"""
import argparse

from nids import data_utils as du
from nids import model


def show(bundle):
    mode = bundle["mode"]
    print("\n" + "=" * 62)
    print(f" {model.MODE_TITLES[mode]}")
    print("=" * 62)
    print(f"Trained on {bundle['train_rows']:,} rows in {bundle['train_seconds']:.1f} s")
    for name in ("validation", "test"):
        m = bundle[name]
        title = "Validation (20% of training data)" if name == "validation" \
            else "Official test set (KDDTest+)"
        print(f"\n--- {title}: accuracy = {m['accuracy'] * 100:.2f}% ---")
        print(m["report_text"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["binary", "multi", "both"], default="both")
    ap.add_argument("--sample", type=int, default=40000,
                    help="training rows to use (0 = all 125,973; slower)")
    ap.add_argument("--compare", action="store_true",
                    help="also compare SVM with other algorithms")
    args = ap.parse_args()

    modes = ["binary", "multi"] if args.mode == "both" else [args.mode]
    for mode in modes:
        bundle = model.train_model(mode, sample_size=args.sample or None)
        show(bundle)
        du.DOCS_DIR.mkdir(exist_ok=True)
        fig = model.plot_confusion_matrix(
            bundle["test"]["confusion"], bundle["labels"],
            f"SVM {mode} - official test set")
        out = du.DOCS_DIR / f"confusion_matrix_{mode}.png"
        fig.savefig(out)
        print(f"Saved {out}")

    if args.compare:
        print("\nComparing algorithms (binary task, official test set) ...")
        table = model.compare_models(sample_size=args.sample or None)
        print(table.round(3).to_string(index=False))
        table.round(3).to_csv(du.DOCS_DIR / "model_comparison.csv", index=False)
