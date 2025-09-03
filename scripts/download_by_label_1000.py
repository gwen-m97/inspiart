import os
import re
import mimetypes
import requests
import pandas as pd
from tqdm import tqdm
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter, Retry

# ======== KONFIG ANPASSEN ========
CSV_PATH   = "../Data/data_sampling1000_topstyles10.csv"  # neue CSV
URL_COL    = "img"            # Spalte mit Bild-URL
NAME_COL   = "file_name"      # Spalte mit Dateinamen (Basisname)
LABEL_COL  = "style"          # Spalte mit Label/Klasse
OUT_ROOT   = "images_1000"    # neuer Zielordner
MAX_PER_LABEL = None          # None = alles aus der CSV laden
TIMEOUT_S  = 15
# =================================

def sanitize(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[^\w\-.]+", "_", s)[:150]

def ensure_ext(filename: str, headers, fallback_from_url: str = "") -> str:
    """
    Falls der Name keine brauchbare Endung hat, nimm Content-Type.
    Wenn der Header unklar ist, versuche Endung aus der URL.
    Fallback: .jpg
    """
    root, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return filename

    ctype = headers.get("Content-Type", "")
    if ctype.startswith("image/"):
        guessed = mimetypes.guess_extension(ctype.split(";")[0].strip())
        if guessed and len(guessed) <= 5:
            return root + guessed

    # Versuche aus URL zu raten
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

    # nur Zeilen mit allen nötigen Feldern
    df = df.dropna(subset=[URL_COL, NAME_COL, LABEL_COL]).copy()

    # pro Label ggf. begrenzen (nur falls du ein Cap setzen willst)
    if MAX_PER_LABEL:
        df = df.groupby(LABEL_COL, group_keys=False).head(MAX_PER_LABEL)

    session = build_session()

    total_ok, total_skip, total_err, total_badctype = 0, 0, 0, 0

    for label, g in df.groupby(LABEL_COL):
        label_dir = os.path.join(OUT_ROOT, sanitize(label))
        os.makedirs(label_dir, exist_ok=True)

        print(f"\n==> {label}  ({len(g)} files)")
        for _, row in tqdm(g.iterrows(), total=len(g), leave=False):
            url  = str(row[URL_COL]).strip()
            name = sanitize(row[NAME_COL]).strip()

            # Zielpfad ohne (oder mit) Dateiendung – wird unten korrigiert
            out_path = os.path.join(label_dir, name)

            # Wenn Datei schon existiert (mit beliebiger gängiger Bild-Endung), skippen
            exists_any = any(os.path.exists(out_path + ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])
            if exists_any or os.path.exists(out_path):
                total_skip += 1
                continue

            try:
                r = session.get(url, timeout=TIMEOUT_S, stream=True)
                if r.status_code == 200:
                    ctype = r.headers.get("Content-Type", "")
                    if not ctype.startswith("image/"):
                        total_badctype += 1
                        # kurze Info, aber nicht spammen
                        # print(f"Nicht-Bild Content-Type ({ctype}) -> {url}")
                        continue

                    out_path_final = ensure_ext(out_path, r.headers, fallback_from_url=url)
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
