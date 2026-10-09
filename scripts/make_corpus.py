"""Bulk English corpus builder: C4/en -> deduped sharded training text.

Downloads allenai/c4 English train shards one at a time (public, no login),
streams them (never full-loads a shard), filters, exact-dedupes via a
disk-backed SQLite store (crash/resume safe), and writes fixed-size text
shards with one document per line::

    python scripts/make_corpus.py --out data/corpus --target-gb 4

Output (``--out DIR``)::

    DIR/shard-00000.txt ...   one whitespace-flattened doc per line
    DIR/dedupe.db             SQLite: seen doc hashes + finished files (resume)
    DIR/manifest.json         source, counts, byte totals, sha256 per shard

"1B tokens" accounting: SmaulBRAIN is byte-level (1 byte ~= 1 token), so
4 GiB ~= 4B byte-tokens ~~= 1B subword tokens (bytes/4). The manifest
reports raw bytes plus the /4 estimate.

Resume: finished source files are skipped via dedupe.db; sharding continues
in the highest existing shard. Safe to Ctrl-C and rerun. Data dirs are
git-ignored (gigabytes do not belong in the repo).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import urllib.error
import urllib.request

REPO = "allenai/c4"
CONFIG = "en"
N_SHARDS = 1024
FILE_TEMPLATE = "en/c4-train.{i:05d}-of-01024.json.gz"

ALLOWED_EXTRA = set(" .,;:!?\'\"()[]{}-–—/\\@#$%^*+=<>|~\n\t")
MIN_KEEP_RATIO = 0.70
MIN_CHARS = 100


def normalize_doc(text: str) -> str:
    """Collapse all whitespace runs to single spaces, strip ends."""
    return " ".join(text.split())


def english_ok(text: str, min_ratio: float = MIN_KEEP_RATIO) -> bool:
    """Mild guard: mostly plain ASCII letters/digits/space/punctuation."""
    if not text:
        return False
    good = sum(1 for c in text
               if c.isascii() and (c.isalnum() or c.isspace() or c in ALLOWED_EXTRA))
    return good / len(text) >= min_ratio


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def open_store(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
    con.execute("CREATE TABLE IF NOT EXISTS done_files (name TEXT PRIMARY KEY)")
    con.execute("PRAGMA journal_mode=WAL")
    return con


def shard_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"shard-{idx:05d}.txt")


class ShardWriter:
    """Append docs, rotating files at shard_bytes. Resumes in last shard."""

    def __init__(self, out_dir: str, shard_bytes: int):
        self.out_dir = out_dir
        self.shard_bytes = shard_bytes
        os.makedirs(out_dir, exist_ok=True)
        idx = 0
        while os.path.exists(shard_path(out_dir, idx + 1)):
            idx += 1
        if os.path.exists(shard_path(out_dir, idx)):
            idx = idx  # continue filling the tail shard
        self.idx = idx
        self.fh = open(shard_path(out_dir, idx), "a", encoding="utf-8")
        self.size = os.path.getsize(shard_path(out_dir, idx))

    def write(self, doc: str) -> str:
        if self.size >= self.shard_bytes:
            self.fh.close()
            self.idx += 1
            self.fh = open(shard_path(self.out_dir, self.idx), "w", encoding="utf-8")
            self.size = 0
        blob = doc + "\n"
        self.fh.write(blob)
        n = len(blob.encode("utf-8"))
        self.size += n
        return shard_path(self.out_dir, self.idx)

    def close(self):
        self.fh.close()


def iter_shard_docs(path: str):
    """Yield raw `text` fields from a C4 json.gz shard, streaming."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)["text"]
            except (ValueError, KeyError, TypeError):
                continue


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _resolve_cdn(url: str) -> str:
    """Follow one redirect manually to the signed CDN URL (fast lane)."""
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        opener.open(urllib.request.Request(url, method="HEAD"), timeout=60)
        return url  # no redirect
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
            return e.headers["Location"]
        raise


def download_shard(i: int, tmp_dir: str, part_dir: str | None = None):
    """Fetch one C4 shard, resumable, via the signed CDN URL.

    Rationale: the /resolve/ endpoint plus the hub client stall on this
    link, while the redirected Xet CDN host moves ~10x faster. Resolve
    once, then ranged-GET the CDN URL with resume into a .part file
    (kept in ``part_dir`` so restarts resume mid-file). Falls back to
    hf_hub_download only if direct fails throughout.
    """
    import time
    import urllib.error
    import urllib.request
    name = FILE_TEMPLATE.format(i=i)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{name}"
    pdir = part_dir or tmp_dir
    os.makedirs(pdir, exist_ok=True)
    dest = os.path.join(tmp_dir, name)
    part = os.path.join(pdir, name + ".part")
    if os.path.exists(dest):
        return name, dest
    try:
        url = _resolve_cdn(url)
    except Exception:
        pass  # fall through: try the original URL directly
    have = os.path.getsize(part) if os.path.exists(part) else 0
    last_err: Exception | None = None
    for attempt in range(12):
        try:
            req = urllib.request.Request(url)
            if have:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=30) as r:
                if have and r.status not in (200, 206):
                    raise IOError(f"unexpected status {r.status}")
                if r.status == 200 and have:
                    have = 0  # server ignored Range: restart
                mode = "ab" if have else "wb"
                with open(part, mode) as f:
                    while True:
                        chunk = r.read(1 << 23)
                        if not chunk:
                            break
                        f.write(chunk)
            os.replace(part, dest)
            return name, dest
        except Exception as e:  # noqa: BLE001 - retry then fallback
            last_err = e
            have = os.path.getsize(part) if os.path.exists(part) else 0
            time.sleep(2 ** attempt)
    try:
        from huggingface_hub import hf_hub_download
        return name, hf_hub_download(REPO, filename=name, repo_type="dataset",
                                     local_dir=tmp_dir)
    except Exception:
        raise RuntimeError(f"direct + hub download failed for {name}: {last_err}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="make_corpus",
                                description="C4/en -> deduped sharded corpus.")
    p.add_argument("--out", default="data/corpus")
    p.add_argument("--target-gb", type=float, default=4.0,
                   help="GiB of kept text to accumulate.")
    p.add_argument("--shard-mb", type=float, default=256.0)
    p.add_argument("--start", type=int, default=0, help="First shard index.")
    p.add_argument("--min-chars", type=int, default=MIN_CHARS)
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan (first file name, targets) without downloading.")
    return p


def run(out: str, target_bytes: int, shard_bytes: int, start: int,
        min_chars: int, log_fn=None) -> dict:
    log = log_fn or (lambda s: print(s, file=sys.stderr, flush=True))
    con = open_store(os.path.join(out, "dedupe.db"))
    writer = ShardWriter(out, shard_bytes)
    stats = {"files": 0, "seen": 0, "kept": 0, "dup": 0, "short": 0,
             "nonenglish": 0, "bytes": 0}
    tmp_dir = tempfile.mkdtemp(prefix="c4dl_")
    try:
        for i in range(start, N_SHARDS):
            if stats["bytes"] >= target_bytes:
                break
            name = FILE_TEMPLATE.format(i=i)
            if con.execute("SELECT 1 FROM done_files WHERE name=?",
                           (name,)).fetchone():
                continue
            log(f"[{i:05d}] downloading {name} ...")
            try:
                _, path = download_shard(i, tmp_dir, part_dir=out)
            except Exception as e:
                log(f"[{i:05d}] download failed, skipping: {e}")
                continue
            batch: list[str] = []
            for raw in iter_shard_docs(path):
                stats["seen"] += 1
                doc = normalize_doc(raw)
                if len(doc) < min_chars:
                    stats["short"] += 1
                    continue
                if not english_ok(doc):
                    stats["nonenglish"] += 1
                    continue
                h = doc_hash(doc)
                batch.append((h, doc))
                if len(batch) >= 2000:
                    _flush(con, writer, batch, stats)
                    batch = []
                    if stats["bytes"] >= target_bytes:
                        break
            _flush(con, writer, batch, stats)
            con.execute("INSERT OR IGNORE INTO done_files VALUES (?)", (name,))
            con.commit()
            try:
                os.remove(path)
            except OSError:
                pass
            stats["files"] += 1
            log(f"[{i:05d}] done: kept={stats['kept']} dup={stats['dup']} "
                f"bytes={stats['bytes'] / 2**30:.2f} GiB")
    finally:
        writer.close()
        con.commit()
        con.close()
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass
    return stats


def _flush(con, writer, batch, stats):
    for h, doc in batch:
        try:
            con.execute("INSERT INTO seen VALUES (?)", (h,))
        except sqlite3.IntegrityError:
            stats["dup"] += 1
            continue
        writer.write(doc)
        stats["kept"] += 1
        stats["bytes"] += len(doc.encode("utf-8")) + 1


def write_manifest(out: str, stats: dict) -> dict:
    shards = []
    idx = 0
    while os.path.exists(shard_path(out, idx)):
        p = shard_path(out, idx)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        shards.append({"path": os.path.basename(p),
                       "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
        idx += 1
    man = {"source": f"{REPO}/{CONFIG}", "stats": stats,
           "bytes_kept": stats["bytes"],
           "est_subword_tokens": stats["bytes"] // 4,
           "shards": shards}
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    target = int(args.target_gb * 2**30)
    if args.dry_run:
        print(f"source={REPO}/{CONFIG} first={FILE_TEMPLATE.format(i=args.start)} "
              f"target={args.target_gb} GiB out={args.out}")
        return 0
    stats = run(args.out, target, int(args.shard_mb * 2**20),
                args.start, args.min_chars)
    man = write_manifest(args.out, stats)
    print(json.dumps({"out": args.out, "bytes_kept": man["bytes_kept"],
                      "est_subword_tokens": man["est_subword_tokens"],
                      "shards": len(man["shards"]), "stats": stats}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
