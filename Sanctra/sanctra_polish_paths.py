# sanctra_polish_paths.py
from __future__ import annotations
from pathlib import Path
import os, re

_HOME = str(Path.home())
_HOME_SLASH = _HOME.replace("\\","/")
_HOME_BACK  = _HOME.replace("/","\\")
_WIN_USER_RE  = re.compile(r"([A-Za-z]:\\Users\\)([^\\\/\n]+)", re.IGNORECASE)
_UNIX_HOME_RE = re.compile(r"(/home/|/Users/)([^/ \n]+)", re.IGNORECASE)

def sanitize_paths(text: str) -> str:
    """Hata/uyarı metinlerinden ev dizinlerini ve uzun mutlak yolları kısaltır."""
    if not isinstance(text, str):
        text = str(text)
    s = text.replace(_HOME, "~").replace(_HOME_SLASH, "~").replace(_HOME_BACK, "~")
    s = _WIN_USER_RE.sub(r"\1…", s)     # C:\Users\metin -> C:\Users\…
    s = _UNIX_HOME_RE.sub(r"\1…", s)    # /Users/metin  -> /Users/…
    s = re.sub(r"([A-Za-z]:\\[^ \n]{10,})", lambda m: _collapse_win(m.group(1)), s)
    s = re.sub(r"(\/[^ \n]{10,})",       lambda m: _collapse_unix(m.group(1)), s)
    return s

def _collapse_win(p: str) -> str:
    parts = p.split("\\"); tail = "\\".join(parts[-2:]) if len(parts) > 2 else p
    return f"…\\{tail}"

def _collapse_unix(p: str) -> str:
    parts = p.split("/"); tail = "/".join(parts[-2:]) if len(parts) > 2 else p
    return f"…/{tail}"

_INVALID = r'<>:"/\\|?*'
def safe_slug(name: str, maxlen: int = 120) -> str:
    s = "".join(("_" if ch in _INVALID else ch) for ch in str(name))
    s = re.sub(r"\s+", "_", s).strip("._ ")
    return (s or "file")[:maxlen]

def default_project_dir() -> Path:
    env = os.environ.get("SANCTRA_DIR")
    if env:
        p = Path(env).expanduser()
    else:
        p = Path("./sanctra_projects")
        try: p.mkdir(parents=True, exist_ok=True)
        except Exception: p = Path.home() / ".sanctra"
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: p = Path.cwd()
    return p

def path_for_download(basename: str, ext: str) -> str:
    """st.download_button için sadece DOSYA ADI döndürür (yol sızdırmaz)."""
    base = safe_slug(basename); ext = ext.lstrip(".")
    return f"{base}.{ext}"

def save_bytes_locally(data: bytes, basename: str, ext: str):
    """İstersen sessizce yerel bir kopya da bırak (UI’da yolu göstermiyoruz)."""
    pdir = default_project_dir()
    fname = path_for_download(basename, ext)
    out = pdir / fname
    out.write_bytes(data)
    return out
