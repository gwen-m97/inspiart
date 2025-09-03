import os
import re
import mimetypes
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter, Retry

# ======== KONFIGURATION anpassen ========
CSV_PATH   = "../Data/data_sampling200_topstyles.csv"  # <— dein CSV-Pfad
URL_COL    = "img"           # Spalte mit Bild-URL
NAME_COL   = "file_name"     # Spalte mit Dateinamen
LABEL_COL  = "style"         #
OUT_ROOT   = "images"        # Zielordner an Repo-Wurzel (laut Screenshot)
MAX_PER_LABEL = None         # z.B. 200 für Limit je Klasse; None = alle
TIMEOUT_S  = 15
# ========================================

def sanitize(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]+", "_", s)[:150]

def ensure_ext(filename: str, headers) -> str:
    root, ext = os.path.splitext(filename)
    if ext.lower() in [".jpg", ".jpeg", ".png", ".webp"]:
        return filename
    ctype = headers.get("Content-Type", "")
    guessed = mimetypes.guess_extension(ctype.split(";")[0].strip()) if ctype else None
    if not guessed or len(guessed) > 5:
        guessed = ".jpg"
    return root + guessed

def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "inspiart-downloader/1.0"})
    return s

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = pd.read_csv(CSV_PATH)
    # nur Zeilen mit allen nötigen Feldern
    df = df.dropna(subset=[URL_COL, NAME_COL, LABEL_COL]).copy()

    # pro Label ggf. begrenzen
    if MAX_PER_LABEL:
        df = (df.groupby(LABEL_COL, group_keys=False)
                .apply(lambda g: g.head(MAX_PER_LABEL)))

    session = build_session()

    total_ok, total_skip, total_err = 0, 0, 0

    for label, g in df.groupby(LABEL_COL):
        label_dir = os.path.join(OUT_ROOT, sanitize(label))
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n==> {label}  ({len(g)} files)")
        for _, row in tqdm(g.iterrows(), total=len(g), leave=False):
            url  = str(row[URL_COL])
            name = sanitize(row[NAME_COL])
            out_path = os.path.join(label_dir, name)

            if os.path.exists(out_path):  # schon vorhanden
                total_skip += 1
                continue

            try:
                r = session.get(url, timeout=TIMEOUT_S, stream=True)
                if r.status_code == 200:
                    # Dateiendung absichern, falls NAME_COL keine/komische hat
                    out_path_final = ensure_ext(out_path, r.headers)
                    with open(out_path_final, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    total_ok += 1
                else:
                    total_err += 1
                    print(f"HTTP {r.status_code} -> {url}")
            except Exception as e:
                total_err += 1
                print(f"ERROR {url}: {e}")

    print(f"\nDone. ok={total_ok} skip={total_skip} err={total_err}")

if __name__ == "__main__":
    main()
