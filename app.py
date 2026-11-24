"""Network Intrusion Detection System - desktop app (Tkinter).

Run:  python app.py
"""
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

from nids import auth, model  # noqa: E402
from nids import data_utils as du  # noqa: E402

BG = "#0f172a"        # dark blue background
PANEL = "#1e293b"
ACCENT = "#38bdf8"
TEXT = "#e2e8f0"
GOOD = "#4ade80"
BAD = "#f87171"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Network Intrusion Detection System - SVM")
        self.geometry("1080x680")
        self.minsize(900, 600)
        self.configure(bg=BG)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", padding=4)

        self.bundles = {}              # loaded models, one per mode
        self.train_df = None
        self.test_df = None
        self.user = None
        self.queue = queue.Queue()     # messages from the training thread
        self.busy = False
        self.show_login()

    # ------------------------------------------------------------ helpers
    def clear(self):
        for widget in self.winfo_children():
            widget.destroy()

    def make_button(self, parent, text, command, color=ACCENT, width=26):
        return tk.Button(parent, text=text, command=command, width=width,
                         bg=color, fg="#0f172a", activebackground="white",
                         font=("Segoe UI", 10, "bold"), relief="flat",
                         cursor="hand2", pady=6)

    # --------------------------------------------------- login / register
    def show_login(self):
        self.clear()
        self.user = None
        card = self._auth_card("Network Intrusion Detection",
                               "Sign in to continue")
        self.username = tk.StringVar()
        self.password = tk.StringVar()
        self._field(card, "Username", self.username)
        self._field(card, "Password", self.password, show="*")
        self.make_button(card, "LOGIN", self.do_login, width=30).pack(pady=(14, 6))
        tk.Button(card, text="Create a new account", command=self.show_register,
                  bg=PANEL, fg=ACCENT, relief="flat", cursor="hand2",
                  font=("Segoe UI", 10, "underline")).pack()
        self.bind("<Return>", lambda e: self.do_login())

    def show_register(self):
        self.clear()
        card = self._auth_card("Create Account", "Passwords are stored hashed")
        self.reg = {k: tk.StringVar() for k in
                    ("full_name", "email", "username", "password", "confirm")}
        self._field(card, "Full name", self.reg["full_name"])
        self._field(card, "E-mail", self.reg["email"])
        self._field(card, "Username", self.reg["username"])
        self._field(card, "Password (6+ chars, letter + digit)",
                    self.reg["password"], show="*")
        self._field(card, "Confirm password", self.reg["confirm"], show="*")
        self.make_button(card, "REGISTER", self.do_register, width=30).pack(pady=(14, 6))
        tk.Button(card, text="Back to login", command=self.show_login,
                  bg=PANEL, fg=ACCENT, relief="flat", cursor="hand2",
                  font=("Segoe UI", 10, "underline")).pack()
        self.bind("<Return>", lambda e: self.do_register())

    def _auth_card(self, title, subtitle):
        outer = tk.Frame(self, bg=BG)
        outer.pack(expand=True)
        tk.Label(outer, text="\U0001F6E1  " + title, bg=BG, fg=TEXT,
                 font=("Segoe UI", 24, "bold")).pack(pady=(0, 4))
        tk.Label(outer, text=subtitle, bg=BG, fg="#94a3b8",
                 font=("Segoe UI", 11)).pack(pady=(0, 16))
        card = tk.Frame(outer, bg=PANEL, padx=30, pady=24)
        card.pack()
        return card

    def _field(self, parent, label, var, show=None):
        tk.Label(parent, text=label, bg=PANEL, fg=TEXT, anchor="w",
                 font=("Segoe UI", 10)).pack(fill="x", pady=(8, 2))
        tk.Entry(parent, textvariable=var, show=show, width=34,
                 font=("Segoe UI", 12), relief="flat").pack(ipady=4)

    def do_login(self):
        ok, msg = auth.login(self.username.get(), self.password.get())
        if ok:
            self.user = self.username.get().strip()
            self.show_dashboard()
        else:
            messagebox.showerror("Login failed", msg)

    def do_register(self):
        r = self.reg
        if r["password"].get() != r["confirm"].get():
            messagebox.showerror("Error", "Passwords do not match.")
            return
        ok, msg = auth.register(r["username"].get(), r["password"].get(),
                                r["full_name"].get(), r["email"].get())
        if ok:
            messagebox.showinfo("Success", msg)
            self.show_login()
        else:
            messagebox.showerror("Error", msg)

    # ----------------------------------------------------------- dashboard
    def show_dashboard(self):
        self.clear()
        self.unbind("<Return>")

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")
        tk.Label(header, text="\U0001F6E1  Network Intrusion Detection (SVM)",
                 bg=PANEL, fg=TEXT, font=("Segoe UI", 16, "bold")
                 ).pack(side="left", padx=16, pady=10)
        tk.Label(header, text=f"Logged in as {self.user}", bg=PANEL,
                 fg="#94a3b8").pack(side="right", padx=16)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        side = tk.Frame(body, bg=BG, padx=14, pady=14)
        side.pack(side="left", fill="y")

        tk.Label(side, text="Detection mode", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.mode_var = tk.StringVar(value=model.MODE_TITLES["binary"])
        combo = ttk.Combobox(side, textvariable=self.mode_var, state="readonly",
                             values=list(model.MODE_TITLES.values()), width=34)
        combo.pack(pady=(2, 14))

        self.buttons = []
        for text, cmd in [
            ("1. Dataset Overview", self.show_overview),
            ("2. Train SVM Model", self.start_training),
            ("3. Show Metrics", self.show_metrics),
            ("4. Confusion Matrix", self.show_confusion),
            ("5. Classify Traffic File (CSV)", self.classify_file),
            ("6. Predict Random Test Sample", self.predict_sample),
        ]:
            b = self.make_button(side, text, cmd, width=32)
            b.pack(pady=4)
            self.buttons.append(b)
        self.make_button(side, "Logout", self.show_login, color=BAD,
                         width=32).pack(pady=(20, 4))

        right = tk.Frame(body, bg=BG, padx=6, pady=14)
        right.pack(side="left", fill="both", expand=True)
        self.output = scrolledtext.ScrolledText(
            right, bg="#020617", fg=TEXT, insertbackground=TEXT,
            font=("Consolas", 10), relief="flat", wrap="word", padx=10, pady=10)
        self.output.pack(fill="both", expand=True, padx=(0, 14))
        self.output.tag_config("good", foreground=GOOD)
        self.output.tag_config("bad", foreground=BAD)
        self.output.tag_config("head", foreground=ACCENT,
                               font=("Consolas", 11, "bold"))

        self.status = tk.StringVar(value="Ready. Start with 'Dataset Overview', then 'Train'.")
        tk.Label(self, textvariable=self.status, bg=PANEL, fg="#94a3b8",
                 anchor="w", padx=12).pack(fill="x", side="bottom")

        self.log("Welcome! Suggested steps:\n"
                 "  1) Dataset Overview  - see what the data looks like\n"
                 "  2) Train SVM Model   - takes a few seconds\n"
                 "  3) Show Metrics / Confusion Matrix\n"
                 "  4) Classify Traffic File - try data/sample_traffic.csv\n",
                 clear=True)
        if not du.data_available():
            self.log("\n! Dataset not found. Run  python download_data.py  "
                     "once, then restart the app.", tag="bad")

    @property
    def mode(self):
        return "binary" if self.mode_var.get() == model.MODE_TITLES["binary"] else "multi"

    def log(self, text, clear=False, tag=None):
        if clear:
            self.output.delete("1.0", "end")
        self.output.insert("end", text + ("\n" if not text.endswith("\n") else ""), tag)
        self.output.see("end")

    def set_busy(self, busy):
        self.busy = busy
        for b in self.buttons:
            b.config(state="disabled" if busy else "normal")

    def need_data(self):
        if not du.data_available():
            messagebox.showwarning("Dataset missing",
                                   "Run  python download_data.py  first.")
            return False
        return True

    def get_bundle(self):
        mode = self.mode
        if mode not in self.bundles:
            self.bundles[mode] = model.load_bundle(mode)
        if self.bundles[mode] is None:
            messagebox.showinfo("No model yet",
                                "Please train the model first (button 2).")
            return None
        return self.bundles[mode]

    # ------------------------------------------------------------ actions
    def show_overview(self):
        if not self.need_data():
            return
        if self.train_df is None:
            self.train_df = du.load_nsl_kdd(du.TRAIN_FILE)
            self.test_df = du.load_nsl_kdd(du.TEST_FILE)
        tr, te = self.train_df, self.test_df
        self.log("DATASET OVERVIEW  (NSL-KDD)", clear=True, tag="head")
        self.log(f"Training set : {len(tr):,} connections\n"
                 f"Test set     : {len(te):,} connections\n"
                 f"Features     : {len(du.FEATURES)} per connection "
                 f"(protocol, service, bytes sent, error rates, ...)\n")
        self.log("Connections per category:", tag="head")
        self.log(f"{'Category':<10}{'Train':>10}{'Test':>10}")
        for c in du.CATEGORY_ORDER:
            self.log(f"{c:<10}{(tr.category == c).sum():>10,}{(te.category == c).sum():>10,}")
        unseen = set(te.label) - set(tr.label)
        self.log(f"\nThe test set contains {len(unseen)} attack types that never appear "
                 "in training - this is why test accuracy is lower than "
                 "validation accuracy.")
        self.log("\nWhat the categories mean:", tag="head")
        self.log("  DoS    - flooding a server so it stops responding\n"
                 "  Probe  - scanning the network for weak points\n"
                 "  R2L    - outsider tries to gain local access (e.g. password guessing)\n"
                 "  U2R    - normal user tries to become administrator")
        self.log("\nFirst 5 rows:", tag="head")
        cols = ["protocol_type", "service", "flag", "src_bytes", "dst_bytes", "label"]
        self.log(tr[cols].head().to_string(index=False))

    def start_training(self):
        if self.busy or not self.need_data():
            return
        mode = self.mode
        self.log(f"TRAINING  -  {model.MODE_TITLES[mode]}", clear=True, tag="head")
        self.set_busy(True)
        self.status.set("Training ... please wait (the window stays responsive).")

        def worker():
            try:
                bundle = model.train_model(
                    mode, progress=lambda m: self.queue.put(("log", m)))
                self.queue.put(("done", bundle))
            except Exception as exc:  # show any error in the GUI
                self.queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self.poll_queue)

    def poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "done":
                    self.bundles[payload["mode"]] = payload
                    self.set_busy(False)
                    self.status.set("Training complete.")
                    self.print_metrics(payload)
                    return
                else:
                    self.set_busy(False)
                    self.status.set("Training failed.")
                    self.log("Error: " + payload, tag="bad")
                    return
        except queue.Empty:
            pass
        self.after(100, self.poll_queue)

    def print_metrics(self, bundle):
        self.log("\nRESULTS  -  " + model.MODE_TITLES[bundle["mode"]], tag="head")
        v, t = bundle["validation"], bundle["test"]
        self.log(f"Validation accuracy (20% of training data): {v['accuracy'] * 100:.2f}%")
        self.log(f"Test accuracy (official KDDTest+)         : {t['accuracy'] * 100:.2f}%\n")
        self.log("Per-class report on the official test set:", tag="head")
        self.log(t["report_text"])
        self.log("Tip: precision = of the alarms raised, how many were real; "
                 "recall = of the real attacks, how many were caught.")

    def show_metrics(self):
        bundle = self.get_bundle()
        if bundle:
            self.log("", clear=True)
            self.print_metrics(bundle)

    def show_confusion(self):
        bundle = self.get_bundle()
        if not bundle:
            return
        win = tk.Toplevel(self)
        win.title("Confusion Matrix (official test set)")
        fig = model.plot_confusion_matrix(
            bundle["test"]["confusion"], bundle["labels"],
            f"SVM - {model.MODE_TITLES[bundle['mode']]}")
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def classify_file(self):
        bundle = self.get_bundle()
        if not bundle:
            return
        path = filedialog.askopenfilename(
            title="Choose a traffic CSV", initialdir=du.DATA_DIR,
            filetypes=[("CSV / text files", "*.csv *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = du.read_traffic_file(path)
            preds = model.predict_dataframe(bundle, df)
        except Exception as exc:
            messagebox.showerror("Could not process file", str(exc))
            return
        df = df.copy()
        df["prediction"] = preds
        self.last_results = df

        self.log(f"CLASSIFICATION  -  {os.path.basename(path)}",
                 clear=True, tag="head")
        self.log(f"Connections analysed: {len(df):,}\n")
        self.log("Predicted distribution:", tag="head")
        for label, count in df["prediction"].value_counts().items():
            self.log(f"  {label:<8}{count:>6,}  ({count / len(df) * 100:.1f}%)")

        if "label" in df.columns:  # file includes the true answer -> check it
            truth = df["label"].map(du.to_category)
            if bundle["mode"] == "binary":
                truth = truth.map(lambda c: "Normal" if c == "Normal" else "Attack")
            acc = (truth.values == df["prediction"].values).mean()
            self.log(f"\nFile contains true labels -> accuracy on this file: "
                     f"{acc * 100:.1f}%", tag="good")

        self.log("\nFirst 15 results:", tag="head")
        cols = ["protocol_type", "service", "src_bytes", "dst_bytes"]
        if "label" in df.columns:
            cols.append("label")
        cols.append("prediction")
        self.log(df[cols].head(15).to_string(index=False))

        if messagebox.askyesno("Save results", "Save all predictions to a CSV file?"):
            out = filedialog.asksaveasfilename(defaultextension=".csv",
                                               initialfile="predictions.csv")
            if out:
                df.to_csv(out, index=False)
                self.status.set(f"Saved {out}")

    def predict_sample(self):
        bundle = self.get_bundle()
        if not bundle or not self.need_data():
            return
        if self.test_df is None:
            self.train_df = du.load_nsl_kdd(du.TRAIN_FILE)
            self.test_df = du.load_nsl_kdd(du.TEST_FILE)
        row = self.test_df.sample(1).iloc[[0]]
        pred = model.predict_dataframe(bundle, row)[0]
        actual = row["category"].iloc[0]
        if bundle["mode"] == "binary":
            actual = "Normal" if actual == "Normal" else "Attack"
        r = row.iloc[0]
        self.log("PREDICT A RANDOM CONNECTION", clear=True, tag="head")
        self.log(f"Protocol / Service / Flag : {r.protocol_type} / {r.service} / {r.flag}\n"
                 f"Bytes sent / received     : {r.src_bytes:,} / {r.dst_bytes:,}\n"
                 f"Connections to same host  : {r['count']}\n"
                 f"SYN error rate            : {r.serror_rate}\n"
                 f"Logged in                 : {'yes' if r.logged_in else 'no'}\n")
        self.log(f"Actual label (from dataset): {actual}   [{r.label}]")
        correct = pred == actual
        self.log(f"Model prediction           : {pred}   "
                 f"{'CORRECT' if correct else 'WRONG'}",
                 tag="good" if correct else "bad")


if __name__ == "__main__":
    App().mainloop()
