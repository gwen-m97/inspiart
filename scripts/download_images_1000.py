import os
import re
import mimetypes
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter, Retry
from pathlib import Path

# ========= PFAD-KONFIG (robust, egal von wo du startest) =========
HERE = Path(__file__).resolve()          # .../inspiart/inspiart/models/download_images_1000.py
ROOT = HERE.parents[1]                   # .../inspiart/inspiart
CSV_PATH = ROOT / "Data" / "data_sampling1000_topstyles10.csv"
OUT_ROOT = ROOT / "images_1000"
# ================================================================

# ========= CSV/SPALTEN-KONFIG =========
URL_COL   = "img"         # Spalte mit Bild-URL
NAME_COL  = "file_name"   # Spalte mit Dateinamen (Basisname)
LABEL_COL = "style"       # Spalte mit Label/Klasse
MAX_PER_LABEL = None      # z.B. 100 für Limit je Klasse; None = alle
TIMEOUT_S = 15
# =====================================

def sanitize(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]+", "_", s)[:150]

def ensure_ext(filename: str, headers, fallback_from_url: str = "") -> str:
    """
    Falls der Name keine brauchbare Endung hat, nutze Content-Type.
    Wenn Header unklar, versuche Endung aus URL. Fallback: .jpg
    """
    root, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return filename

    ctype = headers.get("Content-Type", "")
    if ctype and ctype.startswith("image/"):
        guessed = mimetypes.guess_extension(ctype.split(";")[0].strip())
        if guessed and len(guessed) <= 5:
            return root + guessed

    url_ext = os.path.splitext(urlparse(fallback_from_url).path)[1].lower()
    if url_ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return root + url_ext

    return root + ".jpg"

def build_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "inspiart-downloader/1.1"})
    return s

def download_images():
    os.makedirs(OUT_ROOT, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Nur Zeilen mit allen nötigen Feldern
    df = df.dropna(subset=[URL_COL, NAME_COL, LABEL_COL]).copy()

    # Optional begrenzen
    if MAX_PER_LABEL:
        df = df.groupby(LABEL_COL, group_keys=False).head(MAX_PER_LABEL)

    session = build_session()
    total_ok, total_skip, total_err, total_badctype = 0, 0, 0, 0

    for label, g in df.groupby(LABEL_COL):
        label_dir = OUT_ROOT / sanitize(label)
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n==> {label}  ({len(g)} files)")
        for _, row in tqdm(g.iterrows(), total=len(g), leave=False):
            url = str(row[URL_COL]).strip()
            name = sanitize(row[NAME_COL]).strip()

            # Basis-Zielpfad (Endung wird ggf. ergänzt)
            out_path = label_dir / name

            # Wenn Datei schon existiert (mit gängiger Bild-Endung), skippen
            exists_any = any((out_path.with_suffix(ext)).exists() for ext in [".jpg", ".jpeg", ".png", ".webp"])
            if exists_any or out_path.exists():
                total_skip += 1
                continue

            try:
                r = session.get(url, timeout=TIMEOUT_S, stream=True)
                if r.status_code == 200:
                    ctype = r.headers.get("Content-Type", "")
                    if not ctype or not ctype.startswith("image/"):
                        total_badctype += 1
                        continue

                    # Endung sicherstellen
                    out_path_final = ensure_ext(str(out_path), r.headers, fallback_from_url=url)
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

    print(f"\nDone. ok={total_ok} skip={total_skip} bad-ctype={total_badctype} err={total_err}")

if __name__ == "__main__":
    # Sanity-Checks + Hinweise
    print(f"CSV: {CSV_PATH}")
    print(f"OUT: {OUT_ROOT}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {CSV_PATH}")

    # Spalten prüfen
    _df_head = pd.read_csv(CSV_PATH, nrows=1)
    needed = [URL_COL, NAME_COL, LABEL_COL]
    missing = [c for c in needed if c not in _df_head.columns]
    if missing:
        raise ValueError(f"Spalten fehlen in CSV: {missing}. Gefunden: {list(_df_head.columns)}")

    download_images()
