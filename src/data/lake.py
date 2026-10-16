"""
lake.py - the connection to the deadlock-api.com data lake.

You normally don't need to read this file. It downloads the lake's catalog (a small
file that says which Parquet file holds which matches), connects DuckDB to it, and
knows how each column is stored, so collector.py can build correct SQL.
"""

import json
import re
import urllib.request
from pathlib import Path

import duckdb

MANIFEST_URL = "https://data.deadlock-api.com/v1/manifest.json"

# match_player has ONE ROW PER PLAYER. These columns hold the same value on all
# rows of a match, so they appear once per match in the output.
MATCH_LEVEL = {
    "match_id", "start_time", "duration_s", "match_mode", "game_mode",
    "average_badge_team0", "average_badge_team1", "average_badge",
    "winning_team", "match_outcome", "bot_difficulty",
    "objectives_mask_team0", "objectives_mask_team1",
    "is_high_skill_range_parties", "low_pri_pool", "new_player_pool", "not_scored",
    "game_mode_version", "team_score",
    "match_tracked_stats", "team0_tracked_stats", "team1_tracked_stats",
    "banned_hero_ids", "ranked_type", "rank_interval", "corrupted_penalty_seed",
}
MATCH_LEVEL_PREFIXES = ("objectives.", "mid_boss.", "street_brawl_rounds.")

INT_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
}


class Lake:
    def __init__(self, cache_dir="deadlock_cache", manifest_url=MANIFEST_URL, catalog_path=None):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest = read_json(manifest_url)
        if catalog_path is None:
            catalog_path = download_catalog(manifest, cache_dir)

        self.con = duckdb.connect()
        self.con.execute("INSTALL httpfs; LOAD httpfs; INSTALL ducklake; LOAD ducklake;")
        for setting in ("http_retries = 5", "http_retry_wait_ms = 500", "TimeZone = 'UTC'"):
            try:
                self.con.execute("SET " + setting)
            except duckdb.Error:
                pass  # setting missing in this DuckDB version: harmless
        catalog = "ducklake:" + Path(catalog_path).resolve().as_posix()
        # Tell DuckLake the files live on the web. Without this, DuckLake on Windows
        # turns the "/" in every file URL into "\" and can't open the files.
        data_url = manifest.get("public_url", "https://data.deadlock-api.com").rstrip("/") + "/"
        self.con.execute(
            f"ATTACH {text(catalog)} AS lake "
            f"(READ_ONLY, DATA_PATH {text(data_url)}, OVERRIDE_DATA_PATH true)"
        )

        # A player row can be stored twice (an old copy and a newer one).
        # The *_latest view keeps only the newest, so downloads use it.
        views = {r[0] for r in self.con.execute(
            "SELECT view_name FROM duckdb_views() WHERE database_name = 'lake'").fetchall()}
        self.raw = "lake.match_player"
        self.source = "lake.match_player_latest" if "match_player_latest" in views else self.raw

        self.types = {r[0]: r[1] for r in self.con.execute(f"DESCRIBE {self.raw}").fetchall()}
        info = {c["name"]: c for c in manifest["tables"]["match_player"].get("columns", [])}
        self.ch_types = {n: c.get("type", "") for n, c in info.items()}
        self.comments = {n: c.get("comment") or "" for n, c in info.items()}
        self.enums = {n: parse_enum(t) for n, t in self.ch_types.items() if "Enum" in t}

    def level(self, col):
        """'match' if the column has one value per match, 'player' if one per player."""
        return "match" if col in MATCH_LEVEL or col.startswith(MATCH_LEVEL_PREFIXES) else "player"

    def read(self, col):
        """SQL that reads a column and turns stored numbers into readable values."""
        t = self.types[col]
        if col in self.enums:  # e.g. 4 -> 'Ranked' when the file stores numbers
            cases = " ".join(f"WHEN {k} THEN {text(v)}" for k, v in self.enums[col].items())
            if t.upper() in INT_TYPES:
                return f"CASE {name(col)} {cases} ELSE CAST({name(col)} AS VARCHAR) END"
            if t.endswith("[]") and t[:-2].upper() in INT_TYPES:
                return f"list_transform({name(col)}, lambda x: CASE x {cases} ELSE CAST(x AS VARCHAR) END)"
        if t.upper() in INT_TYPES and "DateTime" in self.ch_types.get(col, ""):
            return f"epoch_ms(CAST({name(col)} AS BIGINT) * 1000)"  # unix seconds -> timestamp
        return name(col)

    def values(self, col, labels):
        """'Ranked', 'unranked'... written the way the column stores them (text or number)."""
        labels = [labels] if isinstance(labels, str) else list(labels)
        enum = self.enums.get(col, {})
        by_label = {v.lower(): (k, v) for k, v in enum.items()}
        out = []
        for label in labels:
            if label.lower() not in by_label:
                options = ", ".join(sorted(v for _, v in by_label.values()))
                raise ValueError(f"Unknown {col} {label!r}. Options: {options}")
            number, proper = by_label[label.lower()]
            out.append(str(number) if self.types[col].upper() in INT_TYPES else text(proper))
        return ", ".join(out)

    def time(self, when):
        """A UTC datetime written to compare directly with start_time."""
        t = self.types["start_time"].upper()
        if t in INT_TYPES:
            return str(int(when.timestamp()))
        if "TIME ZONE" in t:
            return f"TIMESTAMPTZ {text(when.strftime('%Y-%m-%d %H:%M:%S') + '+00')}"
        return f"TIMESTAMP {text(when.strftime('%Y-%m-%d %H:%M:%S'))}"


def read_folder(folder):
    """All day files in a folder, as one pandas DataFrame."""
    files = sorted(p for p in Path(folder).glob("*.parquet") if not p.name.startswith("_"))
    if not files:
        raise FileNotFoundError(f"No .parquet files in {folder}")
    file_list = "[" + ", ".join(text(f.as_posix()) for f in files) + "]"
    return duckdb.sql(
        f"SELECT * FROM read_parquet({file_list}, union_by_name = true) ORDER BY start_time, match_id"
    ).df()


def read_json(url_or_path):
    if str(url_or_path).startswith(("http://", "https://")):
        request = urllib.request.Request(url_or_path, headers={"User-Agent": "deadlock-data"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    return json.loads(Path(url_or_path).read_text())


def download_catalog(manifest, cache_dir):
    """Download the catalog file once per lake version (it changes every hour)."""
    path = cache_dir / f"catalog_v{manifest.get('version', 'latest')}.ducklake"
    if not path.exists():
        url = manifest["public_url"].rstrip("/") + "/" + (manifest.get("catalog") or "v1/catalog.ducklake")
        request = urllib.request.Request(url, headers={"User-Agent": "deadlock-data"})
        part = path.with_name(path.name + ".part")
        with urllib.request.urlopen(request, timeout=300) as response:
            part.write_bytes(response.read())
        part.replace(path)
        for old in cache_dir.glob("catalog_v*.ducklake"):
            if old != path:
                try:
                    old.unlink()
                except OSError:
                    pass  # still open in another program: harmless
    return path


def parse_enum(ch_type):
    """{0: 'Team0', 1: 'Team1', 16: 'Spectator'} from "Enum8('Team0' = 0, ...)"."""
    return {int(v): k for k, v in re.findall(r"'([^']*)'\s*=\s*(-?\d+)", ch_type)}


def name(col):
    """A column name for SQL. Quotes matter: many names contain dots (stats.net_worth)."""
    return '"' + col.replace('"', '""') + '"'


def text(value):
    """A text value for SQL."""
    return "'" + str(value).replace("'", "''") + "'"
