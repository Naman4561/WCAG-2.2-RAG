import os
import re
import time
import hashlib
from urllib.parse import urljoin, urlparse, urldefrag
from typing import Optional
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
START_URLS = [
    "https://www.w3.org/TR/WCAG22/",
    "https://www.w3.org/WAI/WCAG22/Understanding/",
    "https://www.w3.org/WAI/WCAG22/Techniques/",
]


ALLOWED_PREFIXES = tuple(START_URLS)

OUT_DIR = "data/raw"
USER_AGENT = "wcag22-rag-downloader/1.0 (portfolio project; respectful crawl)"
REQUEST_TIMEOUT = 30
SLEEP_SECS = 0.25  

SKIP_SCHEMES = {"mailto", "tel", "javascript"}

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def safe_path_from_url(url: str) -> str:
    """
    Map a URL to a local file path under OUT_DIR.
    Example:
      https://www.w3.org/TR/WCAG22/  -> data/raw/www.w3.org/TR/WCAG22/index.html
      https://www.w3.org/WAI/WCAG22/Understanding/focus-visible.html -> .../focus-visible.html
    """
    parsed = urlparse(url)
    host = parsed.netloc
    path = parsed.path

    if path.endswith("/"):
        path = path + "index.html"

    if not os.path.splitext(path)[1]:
        path = path + ".html"

    local_path = os.path.join(OUT_DIR, host, path.lstrip("/"))
    return local_path


def is_allowed(url: str) -> bool:
    return url.startswith(ALLOWED_PREFIXES)


def normalize_url(base: str, href: str) -> Optional[str]:
    if not href:
        return None
    href = href.strip()

    parsed = urlparse(href)
    if parsed.scheme and parsed.scheme.lower() in SKIP_SCHEMES:
        return None

    absolute = urljoin(base, href)
    absolute, _frag = urldefrag(absolute)

    if not is_allowed(absolute):
        return None

    return absolute


def extract_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")

    links = set()

    for a in soup.find_all("a", href=True):
        u = normalize_url(base_url, a["href"])
        if u:
            links.add(u)

    for tag, attr in [
        ("link", "href"),
        ("script", "src"),
        ("img", "src"),
    ]:
        for node in soup.find_all(tag):
            if node.get(attr):
                u = normalize_url(base_url, node[attr])
                if u:
                    links.add(u)

    return links


def download_url(url: str) -> tuple[bytes, str]:
    """
    Download URL and return (content_bytes, content_type).
    """
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "").lower()
    return resp.content, content_type


def write_file(path: str, content: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(content)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    queue = list(START_URLS)
    seen = set()

    pbar = tqdm(total=0, unit="file")

    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        try:
            content, content_type = download_url(url)
        except Exception as e:
            tqdm.write(f"[WARN] Failed: {url} ({e})")
            continue

        local_path = safe_path_from_url(url)
        write_file(local_path, content)
        pbar.total += 1
        pbar.update(1)

        if "text/html" in content_type or local_path.endswith(".html"):
            try:
                html = content.decode("utf-8", errors="ignore")
                new_links = extract_links(html, url)
                for link in sorted(new_links):
                    if link not in seen:
                        queue.append(link)
            except Exception as e:
                tqdm.write(f"[WARN] Link parse failed: {url} ({e})")

        time.sleep(SLEEP_SECS)

    pbar.close()
    print(f"\nDone. Downloaded {len(seen)} URLs into: {OUT_DIR}")
    print("Tip: Open the local files in your browser to sanity check.")


if __name__ == "__main__":
    main()
