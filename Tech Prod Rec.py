import io
import ast
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import requests
from PIL import Image, ImageTk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ========================== CONFIG ========================== #
DATA_DIR = Path(r"C:\Users\kayod\OneDrive\AI Projects\Tech Product Recommendation\Dataset")
THUMB_SIZE = (260, 260)   # thumbnail size for the details panel


# ======================= DATA UTILITIES ===================== #
CANON_COLS = [
    "title", "category", "description",
    "price", "rating", "image", "link", "source_file"
]

# Use ordered aliases; FORCE image->'image' and link->'link'
ALIASES = {
    "title":      ["title","name","product","product_name","item","item_name"],
    "category":   ["category","main_category","sub_category","dept","department","type","segment"],
    "description":["description","product_description","long_desc","details","specs","main_category","sub_category"],
    "price":      ["discount_price","actual_price","price","amount","list_price","sale_price","current_price"],
    "rating":     ["rating","ratings","stars","review_score","avg_rating"],
    "image":      ["image"],   # <- force ONLY this exact column
    "link":       ["link"],    # <- force ONLY this exact column
}

def normalize_cols(cols):
    return [str(c).strip().lower().replace(" ", "_") for c in cols]

def map_columns(df: pd.DataFrame):
    src = set(df.columns)
    out = {}
    for canon, options in ALIASES.items():
        for opt in options:
            if opt in src:
                out[canon] = opt
                break
    return out

def read_any_csv(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, encoding="utf-8", on_bad_lines="skip")
    df.columns = normalize_cols(df.columns)
    return df

def read_any_excel(fp: Path) -> pd.DataFrame:
    try:
        x = pd.ExcelFile(fp, engine=None)
        frames = [pd.read_excel(x, sheet_name=s) for s in x.sheet_names]
        df = pd.concat(frames, ignore_index=True)
    except Exception:
        df = None
        for eng in (None, "openpyxl", "xlrd"):
            try:
                df = pd.read_excel(fp, engine=eng)
                break
            except Exception:
                continue
        if df is None:
            raise
    df.columns = normalize_cols(df.columns)
    return df

def tidy_and_map(df: pd.DataFrame, source_stem: str) -> pd.DataFrame:
    mapping = map_columns(df)
    if "title" not in mapping:
        raise ValueError("no title column match")

    # rename to canonical
    df = df.rename(columns={v: k for k, v in mapping.items()})

    # Ensure required canon columns exist
    for c in CANON_COLS:
        if c not in df.columns:
            df[c] = None

    # FORCE: only use 'image' col; drop other image-like fields if present
    for c in list(df.columns):
        if c in {"image_url","img","thumbnail","picture"}:
            df.drop(columns=[c], inplace=True, errors="ignore")

    # Force: only use 'link' col; drop other link-like fields if present
    for c in list(df.columns):
        if c in {"url","product_url","page","href"}:
            df.drop(columns=[c], inplace=True, errors="ignore")

    # Clean + coerce
    df["source_file"] = source_stem
    df["title"] = df["title"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).fillna("").str.strip()
    df["description"] = df["description"].fillna("").astype(str)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    return df[CANON_COLS]

def merge_folder(folder: Path) -> pd.DataFrame:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    csvs   = list(folder.glob("*.csv"))
    excels = list(folder.glob("*.xlsx")) + list(folder.glob("*.xls"))
    if not csvs and not excels:
        raise FileNotFoundError("No .csv, .xlsx, or .xls files found in the folder.")

    all_rows = []
    for f in csvs:
        try:
            all_rows.append(tidy_and_map(read_any_csv(f), f.stem))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    for f in excels:
        try:
            all_rows.append(tidy_and_map(read_any_excel(f), f.stem))
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not all_rows:
        raise RuntimeError("No usable files after column mapping.")

    products = pd.concat(all_rows, ignore_index=True)
    products = (
        products.dropna(subset=["title"])
                .drop_duplicates(subset=["title","source_file"], keep="first")
    )
    return products

def build_tfidf(df: pd.DataFrame):
    for c in ["title","category","description"]:
        if c not in df.columns:
            df[c] = ""
    df["combined_text"] = (
        df["title"].fillna("").astype(str) + " " +
        df["category"].fillna("").astype(str) + " " +
        df["description"].fillna("").astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english", min_df=1)
    X = tfidf.fit_transform(df["combined_text"])
    return tfidf, X

def first_url_like(value) -> str:
    """Get the first plausible URL from a cell (handles lists / comma/pipe-separated)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        try:
            arr = ast.literal_eval(s)
            if isinstance(arr, (list, tuple)) and arr:
                return str(arr[0]).strip()
        except Exception:
            pass
    for sep in ["|", ",", ";", " "]:
        if sep in s and "http" in s:
            parts = [p for p in s.split(sep) if "http" in p]
            if parts:
                return parts[0].strip()
    return s


# ======================== TKINTER APP ======================== #
class RecommenderApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tech Product Recommender")
        self.geometry("1360x800")

        # state
        self.products: pd.DataFrame | None = None
        self.X = None
        self.active_indices: list[int] = []
        self.visible_indices: list[int] = []
        self.thumb_cache: dict[str, ImageTk.PhotoImage] = {}
        self.table_iid_to_idx: dict[str, int] = {}

        # ---- top controls
        top = ttk.Frame(self); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text=f"Autoloading from: {DATA_DIR}").grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Reload", command=self.try_autoload_folder).grid(row=0, column=1, padx=8)
        ttk.Separator(top, orient="horizontal").grid(row=1, column=0, columnspan=6, sticky="ew", pady=6)

        # ---- search + k + recommend
        mid = ttk.Frame(self); mid.pack(fill="x", padx=12, pady=8)
        ttk.Label(mid, text="Search in selection:").grid(row=0, column=0, sticky="w")
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(mid, textvariable=self.search_var, width=60)
        self.search_entry.grid(row=0, column=1, padx=6)
        self.search_entry.bind("<KeyRelease>", self.on_search)
        self.search_entry.bind("<Return>", lambda e: self.on_recommend())

        ttk.Label(mid, text="Top-K:").grid(row=0, column=2)
        self.k_var = tk.IntVar(value=10)
        self.k_spin = ttk.Spinbox(mid, from_=5, to=50, textvariable=self.k_var, width=5)
        self.k_spin.grid(row=0, column=3, padx=6)

        ttk.Button(mid, text="Recommend", command=self.on_recommend).grid(row=0, column=4, padx=6)

        # ---- main split: left categories, middle items, right details
        main = ttk.Frame(self); main.pack(fill="both", expand=True, padx=12, pady=8)

        # Left: categories (from source_file)
        left = ttk.Frame(main); left.pack(side="left", fill="y", expand=False)
        ttk.Label(left, text="Categories").pack(anchor="w")
        self.cat_tree = ttk.Treeview(left, show="tree", height=22)
        self.cat_tree.pack(fill="y", expand=False)
        self.cat_tree.bind("<<TreeviewSelect>>", self.on_category_select)

        # Middle: product list + recommendations table
        center = ttk.Frame(main); center.pack(side="left", fill="both", expand=True, padx=(10,10))
        ttk.Label(center, text="Products").pack(anchor="w")
        list_wrap = ttk.Frame(center); list_wrap.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_wrap, height=14, exportselection=False)
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_title)
        self.listbox.bind("<Double-Button-1>", lambda e: self.on_recommend())
        sb_left = ttk.Scrollbar(list_wrap, orient="vertical", command=self.listbox.yview)
        sb_left.pack(side="left", fill="y")
        self.listbox.configure(yscrollcommand=sb_left.set)

        ttk.Label(center, text="Recommendations (within selected category)").pack(anchor="w", pady=(10,0))
        cols = ("title","category","price","rating","source_file","score")
        self.table = ttk.Treeview(center, columns=cols, show="headings", height=12)
        for c in cols:
            self.table.heading(c, text=c)
            self.table.column(c, width=160 if c!="title" else 280, anchor="w")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.on_table_select)
        sb_tbl = ttk.Scrollbar(center, orient="vertical", command=self.table.yview)
        sb_tbl.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=sb_tbl.set)

        # Right: details panel
        right = ttk.LabelFrame(main, text="Details"); right.pack(side="left", fill="y", padx=(10,0))
        self.img_label = ttk.Label(right)
        self.img_label.pack(padx=8, pady=8)

        self.title_var = tk.StringVar(value="")
        self.cat_var = tk.StringVar(value="")
        self.price_var = tk.StringVar(value="")
        self.rating_var = tk.StringVar(value="")
        self.link_url: str | None = None

        ttk.Label(right, textvariable=self.title_var, wraplength=320, font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8)
        ttk.Label(right, textvariable=self.cat_var, wraplength=320).pack(anchor="w", padx=8, pady=(2,0))
        ttk.Label(right, textvariable=self.price_var).pack(anchor="w", padx=8, pady=(6,0))
        ttk.Label(right, textvariable=self.rating_var).pack(anchor="w", padx=8)

        self.link_btn = ttk.Button(right, text="Open Link", command=self.open_link, state="disabled")
        self.link_btn.pack(padx=8, pady=10, anchor="w")

        # ---- status bar
        self.status_var = tk.StringVar(value="Loading dataset...")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w")\
            .pack(fill="x", ipady=2)

        # Autoload
        self.after(200, self.try_autoload_folder)

    # ------------------------ helpers ------------------------ #
    def status(self, text):
        self.status_var.set(text)
        self.update_idletasks()

    def try_autoload_folder(self):
        try:
            df = merge_folder(DATA_DIR)
        except Exception as e:
            self.status(f"Autoload failed: {e}")
            return

        self.init_dataset(df, note=f"Merged {len(df):,} products from {len(df['source_file'].unique())} files.")

    def init_dataset(self, df: pd.DataFrame, note: str):
        for c in CANON_COLS:
            if c not in df.columns:
                df[c] = None

        self.products = df.copy()
        _, self.X = build_tfidf(self.products)

        # build category tree
        self.cat_tree.delete(*self.cat_tree.get_children())
        root_id = self.cat_tree.insert("", "end", text="All", iid="__ALL__")
        for sf in sorted(self.products["source_file"].dropna().unique()):
            self.cat_tree.insert(root_id, "end", text=sf, iid=f"sf::{sf}")
        self.cat_tree.item(root_id, open=True)
        self.cat_tree.selection_set(root_id)

        self.set_active_indices(None)
        self.populate_listbox()

        self.table.delete(*self.table.get_children())
        self.table_iid_to_idx.clear()
        self.clear_details()
        self.status(note + " Model built (TF-IDF). Ready.")

    def on_category_select(self, _event=None):
        sel = self.cat_tree.selection()
        if not sel:
            return
        iid = sel[0]
        if iid == "__ALL__":
            self.set_active_indices(None)
        elif iid.startswith("sf::"):
            self.set_active_indices(iid.split("::", 1)[1])
        else:
            self.set_active_indices(None)
        self.populate_listbox()
        self.table.delete(*self.table.get_children())
        self.table_iid_to_idx.clear()
        self.clear_details()
        self.status("Category selected. Search or pick an item.")

    def set_active_indices(self, source_file: str | None):
        if self.products is None:
            self.active_indices = []
            return
        if source_file is None:
            self.active_indices = list(self.products.index)
        else:
            self.active_indices = list(self.products.index[self.products["source_file"] == source_file])

    def populate_listbox(self, query: str = ""):
        self.listbox.delete(0, tk.END)
        self.visible_indices = []
        if self.products is None:
            return
        titles = self.products.loc[self.active_indices, "title"].astype(str)
        if query:
            mask = titles.str.lower().str.contains(query.lower())
            indices = list(titles[mask].index)
        else:
            indices = list(titles.index)
        titles_sorted = self.products.loc[indices, "title"].astype(str)
        order = titles_sorted.str.lower().argsort(kind="mergesort")
        ordered_idx = [indices[i] for i in order]
        for idx in ordered_idx[:2000]:
            self.listbox.insert(tk.END, self.products.at[idx, "title"])
            self.visible_indices.append(idx)

    def on_search(self, _event=None):
        q = self.search_var.get().strip()
        self.populate_listbox(q)

    def on_recommend(self):
        if self.products is None or self.X is None:
            messagebox.showwarning("Not ready", "Data not loaded yet.")
            return
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("Pick an item", "Select a product in the list.")
            return

        anchor_idx = self.visible_indices[sel[0]]
        sims_vec = cosine_similarity(self.X[anchor_idx], self.X).ravel()
        order = sims_vec.argsort()[::-1]
        active_set = set(self.active_indices)
        k = max(1, min(int(self.k_var.get()), 50))
        rec_idx = [i for i in order if (i != anchor_idx and i in active_set)][:k]

        self.table.delete(*self.table.get_children())
        self.table_iid_to_idx.clear()
        for i in rec_idx:
            row = self.products.iloc[i]
            iid = f"idx::{i}"  # DF index as iid
            self.table.insert("", "end", iid=iid, values=(
                str(row.get("title","")),
                str(row.get("category","")),
                "" if pd.isna(row.get("price")) else str(row.get("price")),
                "" if pd.isna(row.get("rating")) else str(row.get("rating")),
                str(row.get("source_file","")),
                f"{sims_vec[i]:.4f}",
            ))
            self.table_iid_to_idx[iid] = i

        self.status(f"Showing {len(rec_idx)} recommendations.")
        self.show_details(anchor_idx)

    def on_select_title(self, _event=None):
        if not self.visible_indices or self.products is None:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        self.show_details(self.visible_indices[sel[0]])

    def on_table_select(self, _event=None):
        if self.products is None:
            return
        sel = self.table.selection()
        if not sel:
            return
        iid = sel[0]
        idx = self.table_iid_to_idx.get(iid)
        if idx is not None:
            self.show_details(idx)

    def show_details(self, idx: int):
        row = self.products.iloc[idx]
        title = str(row.get("title","")).strip()
        cat   = str(row.get("category","")).strip()
        desc  = str(row.get("description","")).strip()
        price = row.get("price")
        rating = row.get("rating")
        img_url = first_url_like(row.get("image",""))
        link = first_url_like(row.get("link",""))

        self.title_var.set(title)
        cat_text = cat if cat else "(no category)"
        if desc and desc.lower() != cat.lower():
            cat_text += f" • {desc[:120]}{'…' if len(desc)>120 else ''}"
        self.cat_var.set(cat_text)

        self.price_var.set(f"Price: {price:.2f}" if pd.notna(price) else "Price: n/a")
        self.rating_var.set(f"Rating: {rating:.2f}" if pd.notna(rating) else "Rating: n/a")

        self.link_url = link if link else None
        self.link_btn.configure(state="normal" if self.link_url else "disabled")

        self.display_image(img_url)

    def display_image(self, url: str):
        self.img_label.configure(image="", text="")
        if not url:
            self.img_label.configure(text="(no image)")
            return
        if url in self.thumb_cache:
            self.img_label.configure(image=self.thumb_cache[url])
            return
        try:
            if url.startswith("http"):
                r = requests.get(url, timeout=7)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content))
            else:
                img = Image.open(url)
            img.thumbnail(THUMB_SIZE, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumb_cache[url] = photo
            self.img_label.configure(image=photo)
        except Exception:
            self.img_label.configure(text="(image failed to load)")

    def open_link(self):
        if self.link_url:
            try:
                webbrowser.open(self.link_url)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open link:\n{e}")

    def clear_details(self):
        self.title_var.set("")
        self.cat_var.set("")
        self.price_var.set("")
        self.rating_var.set("")
        self.link_url = None
        self.link_btn.configure(state="disabled")
        self.img_label.configure(image="", text="")


if __name__ == "__main__":
    app = RecommenderApp()
    app.mainloop()
