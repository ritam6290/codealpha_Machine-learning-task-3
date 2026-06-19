
import os
import random
import threading
import queue
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


# ============================================================================
# PART 1 — DATA ENGINE (loading, preprocessing, training, prediction)
# ============================================================================

DISEASE_CONFIGS = {
    "Breast Cancer": {
        "filename": "data.csv",
        "target": "diagnosis",
        "positive_label": "M",          # Malignant
        "drop_columns": ["id"],
        "label_type": "categorical",
    },
    "Heart Disease": {
        "filename": "heart.csv",
        "target": "target",
        "positive_label": 1,
        "drop_columns": [],
        "label_type": "binary",
    },
    "Diabetes": {
        "filename": "diabetes.csv",
        "target": "Outcome",
        "positive_label": 1,
        "drop_columns": [],
        "label_type": "binary",
    },
}

MODEL_ORDER = ["Logistic Regression", "SVM", "Random Forest", "XGBoost"]

MODEL_BUILDERS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "SVM": lambda: SVC(kernel="rbf", probability=True, random_state=42),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": lambda: (
        XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss", random_state=42
        ) if HAS_XGBOOST else
        GradientBoostingClassifier(n_estimators=300, random_state=42)
    ),
}


class DiseaseModelBundle:
    """Holds everything needed to predict one disease: scaler, encoder,
    trained models, feature list, and evaluation metrics."""

    def __init__(self, disease_name):
        self.disease_name = disease_name
        self.config = DISEASE_CONFIGS[disease_name]
        self.scaler = StandardScaler()
        self.label_encoder = None
        self.feature_names = []
        self.feature_stats = {}   # name -> dict(min, max, mean, std, is_int_like)
        self.models = {}          # model_name -> fitted estimator
        self.metrics = {}         # model_name -> dict of metrics
        self.roc_data = {}        # model_name -> (fpr, tpr, auc)
        self.best_model_name = None
        self.n_rows = 0
        self.n_features = 0
        self.xgboost_is_fallback = not HAS_XGBOOST

    def load_data(self, folder_path):
        path = os.path.join(folder_path, self.config["filename"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find '{self.config['filename']}' in {folder_path}")

        df = pd.read_csv(path)

        # Drop fully-empty / unnamed junk columns (common in the breast cancer CSV)
        junk_cols = [c for c in df.columns if c.startswith("Unnamed") or df[c].isna().all()]
        df = df.drop(columns=junk_cols, errors="ignore")
        df = df.drop(columns=self.config["drop_columns"], errors="ignore")
        df = df.dropna()

        target_col = self.config["target"]
        y_raw = df[target_col]
        X = df.drop(columns=[target_col])

        if self.config["label_type"] == "categorical":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y_raw)
            pos_idx = list(self.label_encoder.classes_).index(self.config["positive_label"])
            if pos_idx == 0:
                y = 1 - y
        else:
            y = (y_raw == self.config["positive_label"]).astype(int).values

        self.feature_names = list(X.columns)
        self.n_rows, self.n_features = X.shape

        for col in self.feature_names:
            self.feature_stats[col] = {
                "min": float(X[col].min()),
                "max": float(X[col].max()),
                "mean": float(X[col].mean()),
                "std": float(X[col].std()),
                "is_int_like": bool((X[col].dropna() % 1 == 0).all()),
            }

        return X.values, y

    def train(self, folder_path, progress_callback=None):
        X, y = self.load_data(folder_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        best_f1 = -1
        for name, builder in MODEL_BUILDERS.items():
            if progress_callback:
                progress_callback(f"Training {name} on {self.disease_name}...")

            model = builder()
            model.fit(X_train_scaled, y_train)
            self.models[name] = model

            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            auc = None
            fpr, tpr = None, None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                try:
                    auc = roc_auc_score(y_test, y_proba)
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                except Exception:
                    pass

            self.metrics[name] = {
                "accuracy": acc, "precision": prec, "recall": rec,
                "f1": f1, "confusion_matrix": cm, "auc": auc,
            }
            self.roc_data[name] = (fpr, tpr, auc)

            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = name

        return self.metrics

    def predict(self, model_name, feature_values_dict):
        ordered = [feature_values_dict[f] for f in self.feature_names]
        X = np.array(ordered, dtype=float).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        model = self.models[model_name]

        pred = int(model.predict(X_scaled)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_scaled)[0][1])

        return pred, proba


def discover_csv_status(folder_path):
    status = {}
    for disease, cfg in DISEASE_CONFIGS.items():
        status[disease] = os.path.exists(os.path.join(folder_path, cfg["filename"]))
    return status


# ============================================================================
# PART 2 — DESIGN TOKENS
# ============================================================================

class Theme:
    BG = "#0F1B2B"
    BG_PANEL = "#16263B"
    BG_CARD = "#1C3049"
    BG_INPUT = "#22364F"
    BORDER = "#2C415C"

    TEAL = "#2DD4BF"
    TEAL_DARK = "#1A9C8C"
    AMBER = "#F5A623"
    CORAL = "#FF6B6B"
    GREEN = "#4ADE80"

    TEXT_PRIMARY = "#EAF1F7"
    TEXT_SECONDARY = "#8FA3BD"
    TEXT_MUTED = "#5C7290"

    FONT_DISPLAY = ("Georgia", 22, "bold")
    FONT_H1 = ("Segoe UI", 16, "bold")
    FONT_H2 = ("Segoe UI", 12, "bold")
    FONT_BODY = ("Segoe UI", 10)
    FONT_BODY_BOLD = ("Segoe UI", 10, "bold")
    FONT_SMALL = ("Segoe UI", 9)
    FONT_MONO = ("Consolas", 10)
    FONT_BIG_NUM = ("Segoe UI", 28, "bold")


DISEASE_ICONS = {
    "Breast Cancer": "\u2716",
    "Heart Disease": "\u2665",
    "Diabetes": "\u26A1",
}


# ============================================================================
# PART 3 — REUSABLE WIDGETS
# ============================================================================

class Card(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=Theme.BG_CARD, highlightbackground=Theme.BORDER,
                          highlightthickness=1, bd=0, **kwargs)


class PillButton(tk.Button):
    def __init__(self, parent, text, command=None, kind="primary", **kwargs):
        colors = {
            "primary": (Theme.TEAL, "#06201C", Theme.TEAL_DARK),
            "secondary": (Theme.BG_INPUT, Theme.TEXT_PRIMARY, Theme.BORDER),
            "danger": (Theme.CORAL, "#2A0707", "#E55454"),
        }
        bg, fg, active = colors.get(kind, colors["primary"])
        super().__init__(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=active, activeforeground=fg,
            font=Theme.FONT_BODY_BOLD, bd=0, relief="flat",
            padx=18, pady=8, cursor="hand2", **kwargs
        )


class LabeledSlider(tk.Frame):
    """A numeric input row: label, slider, and editable value box kept in sync."""
    def __init__(self, parent, name, lo, hi, default, is_int, on_change=None):
        super().__init__(parent, bg=Theme.BG_PANEL)
        self.var = tk.DoubleVar(value=default)
        self.is_int = is_int
        self.on_change = on_change

        top = tk.Frame(self, bg=Theme.BG_PANEL)
        top.pack(fill="x")
        tk.Label(top, text=name, font=Theme.FONT_BODY, bg=Theme.BG_PANEL,
                  fg=Theme.TEXT_PRIMARY, anchor="w").pack(side="left")

        self.entry = tk.Entry(top, width=8, font=Theme.FONT_BODY, bg=Theme.BG_INPUT,
                               fg=Theme.TEAL, insertbackground=Theme.TEAL, bd=0,
                               justify="right", relief="flat")
        self.entry.pack(side="right", ipady=3)
        self._set_entry_text(default)
        self.entry.bind("<Return>", self._entry_commit)
        self.entry.bind("<FocusOut>", self._entry_commit)

        step = max((hi - lo) / 200, 0.0001)
        self.scale = tk.Scale(
            self, from_=lo, to=hi, orient="horizontal", variable=self.var,
            resolution=(1 if is_int else round(step, 4)),
            showvalue=False, bg=Theme.BG_PANEL, fg=Theme.TEXT_PRIMARY,
            troughcolor=Theme.BG_INPUT, highlightthickness=0, bd=0,
            activebackground=Theme.TEAL, sliderrelief="flat",
            command=self._slider_moved
        )
        self.scale.pack(fill="x", pady=(2, 8))

    def _set_entry_text(self, val):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, str(int(val)) if self.is_int else f"{val:.3f}".rstrip("0").rstrip("."))

    def _slider_moved(self, val):
        self._set_entry_text(float(val))
        if self.on_change:
            self.on_change()

    def _entry_commit(self, event=None):
        try:
            val = float(self.entry.get())
            self.var.set(val)
        except ValueError:
            self._set_entry_text(self.var.get())

    def get(self):
        return self.var.get()


# ============================================================================
# PART 4 — MAIN APPLICATION
# ============================================================================

class MediPredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MediPredict — Multi-Disease Prediction System")
        self.geometry("1180x760")
        self.minsize(1000, 650)
        self.configure(bg=Theme.BG)

        self.bundles = {}
        self.current_disease = None
        self.sliders = {}
        self.train_queue = queue.Queue()

        self._build_layout()
        self._poll_queue()

    # ------------------------------------------------------------------
    def _build_layout(self):
        self._build_header()

        body = tk.Frame(self, bg=Theme.BG)
        body.pack(fill="both", expand=True, padx=24, pady=(0, 20))

        self.setup_frame = tk.Frame(body, bg=Theme.BG)
        self.main_frame = tk.Frame(body, bg=Theme.BG)

        self._build_setup_screen(self.setup_frame)
        self._build_main_screen(self.main_frame)

        self.setup_frame.pack(fill="both", expand=True)

    def _build_header(self):
        header = tk.Frame(self, bg=Theme.BG_PANEL, height=64)
        header.pack(fill="x")
        header.pack_propagate(False)

        inner = tk.Frame(header, bg=Theme.BG_PANEL)
        inner.pack(fill="both", expand=True, padx=24)

        left = tk.Frame(inner, bg=Theme.BG_PANEL)
        left.pack(side="left", fill="y")
        tk.Label(left, text="MediPredict", font=Theme.FONT_DISPLAY,
                  bg=Theme.BG_PANEL, fg=Theme.TEXT_PRIMARY).pack(side="left", pady=10)
        tk.Label(left, text="  multi-disease risk screening", font=Theme.FONT_SMALL,
                  bg=Theme.BG_PANEL, fg=Theme.TEXT_MUTED).pack(side="left", pady=14)

        self.status_dot = tk.Label(inner, text="\u25CF", font=("Segoe UI", 12),
                                     bg=Theme.BG_PANEL, fg=Theme.TEXT_MUTED)
        self.status_dot.pack(side="right", pady=20)
        self.status_label = tk.Label(inner, text="No data loaded", font=Theme.FONT_SMALL,
                                       bg=Theme.BG_PANEL, fg=Theme.TEXT_SECONDARY)
        self.status_label.pack(side="right", padx=(0, 8), pady=20)

    # ------------------------------------------------------------------
    # SETUP SCREEN
    # ------------------------------------------------------------------
    def _build_setup_screen(self, parent):
        wrap = tk.Frame(parent, bg=Theme.BG)
        wrap.place(relx=0.5, rely=0.42, anchor="center")

        tk.Label(wrap, text="Point me to your data folder",
                  font=("Segoe UI", 20, "bold"), bg=Theme.BG,
                  fg=Theme.TEXT_PRIMARY).pack(pady=(0, 6))
        tk.Label(wrap, text="The folder should contain data.csv, heart.csv, and diabetes.csv",
                  font=Theme.FONT_BODY, bg=Theme.BG, fg=Theme.TEXT_SECONDARY).pack(pady=(0, 24))

        path_card = Card(wrap, padx=4, pady=4)
        path_card.pack()
        self.path_var = tk.StringVar(value=r"C:\Users\ritam\Downloads\health")
        self.path_entry = tk.Entry(path_card, textvariable=self.path_var, width=52,
                                     font=Theme.FONT_MONO, bg=Theme.BG_INPUT,
                                     fg=Theme.TEXT_PRIMARY, insertbackground=Theme.TEAL,
                                     bd=0, relief="flat")
        self.path_entry.pack(side="left", ipady=10, ipadx=10, padx=(8, 4))
        browse_btn = PillButton(path_card, "Browse\u2026", kind="secondary",
                                  command=self._browse_folder)
        browse_btn.pack(side="left", padx=4)

        self.train_btn = PillButton(wrap, "Load & Train Models", kind="primary",
                                      command=self._start_training)
        self.train_btn.pack(pady=20, ipadx=6)

        self.csv_status_frame = tk.Frame(wrap, bg=Theme.BG)
        self.csv_status_frame.pack(pady=(4, 0))
        self.csv_status_labels = {}
        for disease in DISEASE_CONFIGS:
            row = tk.Frame(self.csv_status_frame, bg=Theme.BG)
            row.pack(anchor="w", pady=2)
            dot = tk.Label(row, text="\u25CB", font=Theme.FONT_BODY, bg=Theme.BG,
                             fg=Theme.TEXT_MUTED, width=2)
            dot.pack(side="left")
            lbl = tk.Label(row, text=f"{disease}  ({DISEASE_CONFIGS[disease]['filename']})",
                             font=Theme.FONT_SMALL, bg=Theme.BG, fg=Theme.TEXT_MUTED)
            lbl.pack(side="left")
            self.csv_status_labels[disease] = (dot, lbl)

        self.progress_var = tk.StringVar(value="")
        self.progress_label = tk.Label(wrap, textvariable=self.progress_var,
                                         font=Theme.FONT_SMALL, bg=Theme.BG,
                                         fg=Theme.TEAL, wraplength=520)
        self.progress_label.pack(pady=(16, 0))

        if not HAS_MATPLOTLIB:
            tk.Label(wrap, text="(matplotlib not found — charts will be skipped, everything else still works)",
                      font=Theme.FONT_SMALL, bg=Theme.BG, fg=Theme.AMBER).pack(pady=(8, 0))

        self.path_var.trace_add("write", lambda *a: self._refresh_csv_status())
        self._refresh_csv_status()

    def _refresh_csv_status(self):
        folder = self.path_var.get().strip()
        if not folder or not os.path.isdir(folder):
            for dot, lbl in self.csv_status_labels.values():
                dot.config(text="\u25CB", fg=Theme.TEXT_MUTED)
            return
        status = discover_csv_status(folder)
        for disease, found in status.items():
            dot, lbl = self.csv_status_labels[disease]
            if found:
                dot.config(text="\u25CF", fg=Theme.GREEN)
            else:
                dot.config(text="\u25CF", fg=Theme.CORAL)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing the CSV files")
        if folder:
            self.path_var.set(folder)

    def _start_training(self):
        folder = self.path_var.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Folder not found",
                                  f"Couldn't find this folder:\n{folder}\n\n"
                                  "Please check the path and try again.")
            return

        status = discover_csv_status(folder)
        missing = [d for d, found in status.items() if not found]
        if missing:
            names = ", ".join(f"{d} ({DISEASE_CONFIGS[d]['filename']})" for d in missing)
            messagebox.showerror("Missing files",
                                  f"These CSV files are missing from the folder:\n{names}")
            return

        self.train_btn.config(state="disabled", text="Training\u2026")
        self.path_entry.config(state="disabled")
        threading.Thread(target=self._train_worker, args=(folder,), daemon=True).start()

    def _train_worker(self, folder):
        try:
            for disease in DISEASE_CONFIGS:
                bundle = DiseaseModelBundle(disease)
                bundle.train(folder, progress_callback=lambda msg: self.train_queue.put(("progress", msg)))
                self.bundles[disease] = bundle
                self.train_queue.put(("disease_done", disease))
            self.train_queue.put(("all_done", None))
        except Exception as e:
            self.train_queue.put(("error", str(e)))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.train_queue.get_nowait()
                if kind == "progress":
                    self.progress_var.set(payload)
                elif kind == "disease_done":
                    pass
                elif kind == "all_done":
                    self._on_training_complete()
                elif kind == "error":
                    self.train_btn.config(state="normal", text="Load & Train Models")
                    self.path_entry.config(state="normal")
                    messagebox.showerror("Training failed", payload)
        except queue.Empty:
            pass
        self.after(120, self._poll_queue)

    def _on_training_complete(self):
        self.progress_var.set("All models trained successfully.")
        self.status_dot.config(fg=Theme.GREEN)
        self.status_label.config(text=f"{len(self.bundles)} datasets trained")

        self.setup_frame.pack_forget()
        self.main_frame.pack(fill="both", expand=True)
        first = list(DISEASE_CONFIGS.keys())[0]
        self._select_disease(first)

    # ------------------------------------------------------------------
    # MAIN SCREEN
    # ------------------------------------------------------------------
    def _build_main_screen(self, parent):
        nav = tk.Frame(parent, bg=Theme.BG_PANEL, width=220)
        nav.pack(side="left", fill="y")
        nav.pack_propagate(False)

        tk.Label(nav, text="DATASETS", font=Theme.FONT_SMALL, bg=Theme.BG_PANEL,
                  fg=Theme.TEXT_MUTED).pack(anchor="w", padx=20, pady=(20, 8))

        self.nav_buttons = {}
        for disease in DISEASE_CONFIGS:
            btn = tk.Button(
                nav, text=f"  {DISEASE_ICONS.get(disease,'')}   {disease}",
                font=Theme.FONT_BODY_BOLD, bg=Theme.BG_PANEL, fg=Theme.TEXT_SECONDARY,
                activebackground=Theme.BG_CARD, activeforeground=Theme.TEAL,
                bd=0, anchor="w", padx=14, pady=12, cursor="hand2",
                command=lambda d=disease: self._select_disease(d)
            )
            btn.pack(fill="x", padx=8, pady=2)
            self.nav_buttons[disease] = btn

        ttk.Separator(nav, orient="horizontal").pack(fill="x", padx=16, pady=16)
        retrain_btn = PillButton(nav, "\u21BB  Retrain / New Folder", kind="secondary",
                                   command=self._back_to_setup)
        retrain_btn.pack(padx=12, fill="x")

        content = tk.Frame(parent, bg=Theme.BG)
        content.pack(side="left", fill="both", expand=True, padx=(20, 0))

        canvas = tk.Canvas(content, bg=Theme.BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(content, orient="vertical", command=canvas.yview)
        self.scroll_inner = tk.Frame(canvas, bg=Theme.BG)

        self.scroll_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_inner, anchor="nw", width=900)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self.content_canvas = canvas

        self.section_title = tk.Label(self.scroll_inner, text="", font=Theme.FONT_DISPLAY,
                                        bg=Theme.BG, fg=Theme.TEXT_PRIMARY)
        self.section_title.pack(anchor="w", pady=(20, 4))
        self.section_subtitle = tk.Label(self.scroll_inner, text="", font=Theme.FONT_BODY,
                                           bg=Theme.BG, fg=Theme.TEXT_SECONDARY)
        self.section_subtitle.pack(anchor="w", pady=(0, 16))

        self.compare_card = Card(self.scroll_inner)
        self.compare_card.pack(fill="x", pady=(0, 16))

        self.work_area = tk.Frame(self.scroll_inner, bg=Theme.BG)
        self.work_area.pack(fill="x", pady=(0, 30))

        self.form_card = Card(self.work_area)
        self.form_card.pack(side="left", fill="both", expand=True, padx=(0, 16))

        self.result_card = Card(self.work_area, width=320)
        self.result_card.pack(side="left", fill="y")
        self.result_card.pack_propagate(False)

    def _back_to_setup(self):
        self.main_frame.pack_forget()
        self.setup_frame.pack(fill="both", expand=True)
        self.train_btn.config(state="normal", text="Load & Train Models")
        self.path_entry.config(state="normal")

    # ------------------------------------------------------------------
    def _select_disease(self, disease):
        self.current_disease = disease
        for d, btn in self.nav_buttons.items():
            if d == disease:
                btn.config(bg=Theme.BG_CARD, fg=Theme.TEAL)
            else:
                btn.config(bg=Theme.BG_PANEL, fg=Theme.TEXT_SECONDARY)

        bundle = self.bundles[disease]
        self.section_title.config(text=f"{DISEASE_ICONS.get(disease,'')}  {disease}")
        fallback_note = "  (XGBoost unavailable \u2014 used Gradient Boosting instead)" if bundle.xgboost_is_fallback else ""
        self.section_subtitle.config(
            text=f"{bundle.n_rows} patient records \u00B7 {bundle.n_features} features \u00B7 "
                 f"best model: {bundle.best_model_name}{fallback_note}"
        )

        self._build_comparison_section(bundle)
        self._build_prediction_form(bundle)
        self._clear_result_card(bundle)

    # ------------------------------------------------------------------
    # MODEL COMPARISON SECTION
    # ------------------------------------------------------------------
    def _build_comparison_section(self, bundle):
        for w in self.compare_card.winfo_children():
            w.destroy()

        tk.Label(self.compare_card, text="Model comparison", font=Theme.FONT_H2,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=20, pady=(16, 10))

        table = tk.Frame(self.compare_card, bg=Theme.BG_CARD)
        table.pack(fill="x", padx=20, pady=(0, 8))
        headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
        widths = [20, 10, 10, 10, 10, 10]
        for h, w in zip(headers, widths):
            tk.Label(table, text=h, font=Theme.FONT_SMALL, bg=Theme.BG_CARD,
                      fg=Theme.TEXT_MUTED, width=w, anchor="w").grid(row=0, column=headers.index(h), sticky="w")

        for i, model_name in enumerate(MODEL_ORDER, start=1):
            m = bundle.metrics[model_name]
            is_best = model_name == bundle.best_model_name
            fg = Theme.TEAL if is_best else Theme.TEXT_PRIMARY
            name_text = f"\u2605 {model_name}" if is_best else f"   {model_name}"
            vals = [name_text, f"{m['accuracy']:.1%}", f"{m['precision']:.1%}",
                    f"{m['recall']:.1%}", f"{m['f1']:.1%}",
                    f"{m['auc']:.3f}" if m['auc'] is not None else "\u2014"]
            for j, (v, w) in enumerate(zip(vals, widths)):
                tk.Label(table, text=v, font=Theme.FONT_BODY, bg=Theme.BG_CARD,
                          fg=fg, width=w, anchor="w").grid(row=i, column=j, sticky="w", pady=3)

        if HAS_MATPLOTLIB:
            self._build_charts(bundle)
        else:
            tk.Label(self.compare_card,
                      text="(install matplotlib to see accuracy/ROC charts here)",
                      font=Theme.FONT_SMALL, bg=Theme.BG_CARD,
                      fg=Theme.TEXT_MUTED).pack(anchor="w", padx=20, pady=(0, 16))

    def _build_charts(self, bundle):
        chart_frame = tk.Frame(self.compare_card, bg=Theme.BG_CARD)
        chart_frame.pack(fill="x", padx=12, pady=(4, 16))

        fig = Figure(figsize=(9.0, 2.6), dpi=100, facecolor=Theme.BG_CARD)

        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_facecolor(Theme.BG_CARD)
        names = MODEL_ORDER
        accs = [bundle.metrics[n]["accuracy"] for n in names]
        colors = [Theme.TEAL if n == bundle.best_model_name else Theme.TEXT_MUTED for n in names]
        bars = ax1.bar(range(len(names)), accs, color=colors)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=7, color=Theme.TEXT_SECONDARY)
        ax1.set_ylim(0, 1.05)
        ax1.set_title("Accuracy", fontsize=9, color=Theme.TEXT_PRIMARY)
        ax1.tick_params(axis='y', colors=Theme.TEXT_MUTED, labelsize=7)
        for spine in ax1.spines.values():
            spine.set_color(Theme.BORDER)
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.0%}",
                      ha="center", fontsize=7, color=Theme.TEXT_PRIMARY)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor(Theme.BG_CARD)
        for n in names:
            fpr, tpr, auc = bundle.roc_data[n]
            if fpr is not None:
                lw = 2.4 if n == bundle.best_model_name else 1.2
                alpha = 1.0 if n == bundle.best_model_name else 0.5
                ax2.plot(fpr, tpr, label=f"{n}", linewidth=lw, alpha=alpha)
        ax2.plot([0, 1], [0, 1], linestyle="--", color=Theme.TEXT_MUTED, linewidth=0.8)
        ax2.set_title("ROC Curves", fontsize=9, color=Theme.TEXT_PRIMARY)
        ax2.tick_params(colors=Theme.TEXT_MUTED, labelsize=7)
        ax2.legend(fontsize=6, facecolor=Theme.BG_CARD, edgecolor=Theme.BORDER, labelcolor=Theme.TEXT_SECONDARY)
        for spine in ax2.spines.values():
            spine.set_color(Theme.BORDER)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="x")

    # ------------------------------------------------------------------
    # PREDICTION FORM
    # ------------------------------------------------------------------
    def _build_prediction_form(self, bundle):
        for w in self.form_card.winfo_children():
            w.destroy()
        self.sliders = {}

        header = tk.Frame(self.form_card, bg=Theme.BG_CARD)
        header.pack(fill="x", padx=20, pady=(16, 4))
        tk.Label(header, text="Patient input", font=Theme.FONT_H2,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(side="left")

        tk.Label(header, text="Model:", font=Theme.FONT_SMALL, bg=Theme.BG_CARD,
                  fg=Theme.TEXT_SECONDARY).pack(side="left", padx=(30, 6))
        self.model_var = tk.StringVar(value=bundle.best_model_name)
        model_dropdown = ttk.Combobox(header, textvariable=self.model_var, values=MODEL_ORDER,
                                        state="readonly", width=18, font=Theme.FONT_SMALL)
        model_dropdown.pack(side="left")

        randomize_btn = PillButton(header, "\u21BB Random sample", kind="secondary",
                                     command=lambda: self._fill_random_sample(bundle))
        randomize_btn.pack(side="right")

        scroll_wrap = tk.Frame(self.form_card, bg=Theme.BG_PANEL)
        scroll_wrap.pack(fill="both", expand=True, padx=20, pady=16)

        form_canvas = tk.Canvas(scroll_wrap, bg=Theme.BG_PANEL, highlightthickness=0, height=380)
        form_scroll = ttk.Scrollbar(scroll_wrap, orient="vertical", command=form_canvas.yview)
        inner = tk.Frame(form_canvas, bg=Theme.BG_PANEL)
        inner.bind("<Configure>", lambda e: form_canvas.configure(scrollregion=form_canvas.bbox("all")))
        form_canvas.create_window((0, 0), window=inner, anchor="nw", width=560)
        form_canvas.configure(yscrollcommand=form_scroll.set)
        form_canvas.pack(side="left", fill="both", expand=True)
        form_scroll.pack(side="right", fill="y")

        for feat in bundle.feature_names:
            stats = bundle.feature_stats[feat]
            lo = stats["min"] if stats["min"] != stats["max"] else stats["min"] - 1
            hi = stats["max"] if stats["min"] != stats["max"] else stats["max"] + 1
            default = round(stats["mean"], 3)
            slider = LabeledSlider(inner, self._friendly_name(feat), lo, hi, default,
                                     stats["is_int_like"])
            slider.pack(fill="x", padx=4, pady=4)
            self.sliders[feat] = slider

        predict_btn = PillButton(self.form_card, "Predict", kind="primary",
                                   command=lambda: self._run_prediction(bundle))
        predict_btn.pack(pady=(0, 18), ipadx=20)

    def _friendly_name(self, raw):
        return raw.replace("_", " ").strip().title()

    def _fill_random_sample(self, bundle):
        for feat, slider in self.sliders.items():
            stats = bundle.feature_stats[feat]
            lo, hi = stats["min"], stats["max"]
            val = random.uniform(lo, hi)
            if stats["is_int_like"]:
                val = round(val)
            slider.var.set(val)
            slider._set_entry_text(val)

    # ------------------------------------------------------------------
    # RESULT CARD
    # ------------------------------------------------------------------
    def _clear_result_card(self, bundle):
        for w in self.result_card.winfo_children():
            w.destroy()
        tk.Label(self.result_card, text="Prediction", font=Theme.FONT_H2,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=20, pady=(16, 6))
        tk.Label(self.result_card, text="Set the patient values on the left,\nthen click Predict.",
                  font=Theme.FONT_BODY, bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
                  justify="left").pack(anchor="w", padx=20, pady=(0, 16))

    def _run_prediction(self, bundle):
        values = {feat: slider.get() for feat, slider in self.sliders.items()}
        model_name = self.model_var.get()
        pred, proba = bundle.predict(model_name, values)

        for w in self.result_card.winfo_children():
            w.destroy()

        tk.Label(self.result_card, text="Prediction", font=Theme.FONT_H2,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=20, pady=(16, 10))

        positive = pred == 1
        accent = Theme.CORAL if positive else Theme.GREEN
        verdict_text = "Disease likely" if positive else "Disease unlikely"
        icon = "\u26A0" if positive else "\u2713"

        badge = tk.Frame(self.result_card, bg=Theme.BG_CARD)
        badge.pack(anchor="w", padx=20, pady=(0, 4))
        tk.Label(badge, text=icon, font=("Segoe UI", 22), bg=Theme.BG_CARD, fg=accent).pack(side="left")
        tk.Label(badge, text=verdict_text, font=Theme.FONT_H1, bg=Theme.BG_CARD, fg=accent).pack(side="left", padx=8)

        if proba is not None:
            tk.Label(self.result_card, text=f"{proba:.1%}", font=Theme.FONT_BIG_NUM,
                      bg=Theme.BG_CARD, fg=accent).pack(anchor="w", padx=20, pady=(8, 0))
            tk.Label(self.result_card, text="estimated probability", font=Theme.FONT_SMALL,
                      bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w", padx=20, pady=(0, 12))

            bar_bg = tk.Frame(self.result_card, bg=Theme.BG_INPUT, height=10, width=260)
            bar_bg.pack(anchor="w", padx=20, pady=(0, 16))
            bar_bg.pack_propagate(False)
            fill_w = max(int(260 * proba), 2)
            tk.Frame(bar_bg, bg=accent, height=10, width=fill_w).place(x=0, y=0)

        ttk.Separator(self.result_card, orient="horizontal").pack(fill="x", padx=20, pady=8)

        tk.Label(self.result_card, text="Model used", font=Theme.FONT_SMALL,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED).pack(anchor="w", padx=20, pady=(8, 0))
        tk.Label(self.result_card, text=model_name, font=Theme.FONT_BODY_BOLD,
                  bg=Theme.BG_CARD, fg=Theme.TEXT_PRIMARY).pack(anchor="w", padx=20)

        m = bundle.metrics[model_name]
        tk.Label(self.result_card, text=f"Model accuracy on test data: {m['accuracy']:.1%}",
                  font=Theme.FONT_SMALL, bg=Theme.BG_CARD, fg=Theme.TEXT_SECONDARY,
                  wraplength=260, justify="left").pack(anchor="w", padx=20, pady=(8, 0))

        tk.Label(self.result_card,
                  text="This is a statistical screening estimate, not a medical\ndiagnosis. Consult a clinician.",
                  font=Theme.FONT_SMALL, bg=Theme.BG_CARD, fg=Theme.TEXT_MUTED,
                  wraplength=260, justify="left").pack(anchor="w", padx=20, pady=(18, 16), side="bottom")


def main():
    app = MediPredictApp()
    app.mainloop()


if __name__ == "__main__":
    main()
