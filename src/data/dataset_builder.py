"""
dataset_builder.py - build a dataset of matches and save it on your PC.

    from deadlock_data import DatasetBuilder

    # Training data: every match before July 30 (only ranked existed then),
    # unranked matches only from July 30 on
    DatasetBuilder("data/train").build(start="2026-04-23", end="2026-10-04",
                                       include_all_before="2026-07-30")

    # Test data: ranked matches, Aug 1 to Oct 4 (end is not included)
    DatasetBuilder("data/ranked_test").build(start="2026-08-01", end="2026-10-05", match_mode="Ranked")

Both share the raw downloads in data/players, so days needed by both are
downloaded once. One row per match, oldest first, saved in the folder as:
    inputs.npy        63 integers per match (names in info.json)
    outputs.npy        6 integers per match
    rank_tiers.npy    rank bucket of each match: 1 = Initiate ... 11 = Eternus, 0 = unknown
    rank_scores.npy   the average rank behind it, on a 1-66 scale (6 steps per rank)
    match_ids.npy     which match each row is
    start_times.npy   when it started (unix seconds, UTC)
    info.json         column names, rank names, dates
Read it back with TrainingData (training_data.py).

Inputs, per match (t0 = side 0, t1 = side 1, p1..p6 = players ordered by slot):
    hero_*   12  hero id
    lane_*   12  the game's lane number (0 = unknown)
    party_*  12  party size: 1 = solo, 2 = duo, ...
    games_*  12  games this player had on this hero BEFORE this match
    wins_*   12  wins this player had on this hero BEFORE this match
    3 pool flags (0/1): new_player_pool, low_pri_pool, high_skill_range_parties
Outputs, per match:
    team0_won (0/1), duration_s, kills_team0, kills_team1,
    midbosses_team0, midbosses_team1 (mid-bosses CLAIMED by each side)

Rank of a match = the average rank the players entered it with. If no player
rank is recorded, the average of the two teams' average badges is used instead.

Experience is counted from every normal match (ranked and unranked) in the span
plus `history_days` of warm-up before it. Matches where a player left, or a side
doesn't have 6 players, are left out.
"""

import datetime as dt
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .lake import Lake, name, text

RANK_NAMES = ["Initiate", "Seeker", "Acolyte", "Sentinel", "Mystic", "Ritualist",
              "Emissary", "Oracle", "Phantom", "Ascendant", "Eternus"]

SLOTS = [f"t{team}p{player}" for team in (0, 1) for player in range(1, 7)]
PER_PLAYER = [  # (column built below, name prefix in the dataset)
    ("hero_id", "hero"), ("lane", "lane"), ("party_size", "party"),
    ("games_before", "games"), ("wins_before", "wins"),
]
FLAGS = ["new_player_pool", "low_pri_pool", "is_high_skill_range_parties"]
INPUT_NAMES = [f"{prefix}_{slot}" for _, prefix in PER_PLAYER for slot in SLOTS] + [
    "new_player_pool", "low_pri_pool", "high_skill_range_parties"]
OUTPUT_NAMES = ["team0_won", "duration_s", "kills_team0", "kills_team1",
                "midbosses_team0", "midbosses_team1"]

# What gets downloaded for every player of every normal match.
BASE_COLUMNS = [
    "match_id", "start_time", "match_mode", "duration_s", "winning_team", *FLAGS,
    "mid_boss.team_claimed",
    "account_id", "team", "player_slot", "hero_id", "won", "assigned_lane", "party",
    "kills", "abandon_match_time_s",
]
RANK_COLUMNS = ["average_badge_team0", "average_badge_team1", "player_rank_initial_display_rank"]
DOWNLOAD_COLUMNS = BASE_COLUMNS + RANK_COLUMNS
BYTES_PER_MATCH = (len(INPUT_NAMES) + len(OUTPUT_NAMES)) * 2 + 1 + 4 + 16  # int16 values, rank, ids
DATASET_VERSION = 2  # bump when the saved format changes, so old datasets get rebuilt
DATASET_FILES = ["inputs.npy", "outputs.npy", "rank_tiers.npy", "rank_scores.npy",
                 "match_ids.npy", "start_times.npy", "info.json"]


def ensure_dataset(folder, start, end=None, match_mode="Unranked", include_all_before=None,
                   history_days=90, players_dir="data/players", ask=True, **lake_options):
    """Use the dataset in `folder` if it was already built with these settings;
    otherwise build it (asking Y/N first). Returns the folder, or None if you said N.

        folder = ensure_dataset("data/train", start="2026-04-23", end="2026-10-04",
                                include_all_before="2026-07-30")
    """
    folder = Path(folder)
    days = _days(start, end)
    wanted = {
        "version": DATASET_VERSION,
        "match_mode": match_mode.capitalize(),
        "include_all_before": str(_date(include_all_before)) if include_all_before else None,
        "start": str(days[0]) if days else None,
        "end": str(days[-1]) if days else None,
        "history_days": history_days,
    }
    info_file = folder / "info.json"
    if info_file.exists() and all((folder / f).exists() for f in DATASET_FILES):
        info = json.loads(info_file.read_text())
        different = [f"{k} was {info.get(k)}, now {v}" for k, v in wanted.items() if info.get(k) != v]
        if not different:
            print(f"Using the dataset already in {folder} ({info['matches']:,} matches).")
            return folder
        print(f"{folder} was built with other settings ({'; '.join(different)}); building it again.")
    else:
        print(f"No finished dataset in {folder}; building it.")
    return DatasetBuilder(folder, players_dir=players_dir, **lake_options).build(
        start, end, match_mode=match_mode, include_all_before=include_all_before,
        history_days=history_days, ask=ask)


class DatasetBuilder:
    def __init__(self, folder="data/dataset", players_dir="data/players",
                 cache_dir="deadlock_cache", **lake_options):
        self.folder = Path(folder)
        self.days_dir = Path(players_dir)  # raw downloads, shared between datasets
        print("Connecting to the data lake...")
        self.lake = Lake(cache_dir, **lake_options)
        missing = [c for c in DOWNLOAD_COLUMNS if c not in self.lake.types]
        if missing:
            raise ValueError(f"The data lake has no column(s) {missing}")

    def build(self, start, end=None, match_mode="Unranked", include_all_before=None,
              history_days=90, ask=True):
        """Estimate, ask, download, then build. Safe to stop and re-run: days already
        downloaded are skipped. Returns the folder, or None if cancelled.

        include_all_before: a date like "2026-07-30". Matches before it are kept
        whatever their mode (ranked or unranked); from that date on, only
        `match_mode` matches are kept."""
        if match_mode.lower() not in ("ranked", "unranked"):
            raise ValueError("match_mode must be 'Ranked' or 'Unranked'")
        match_mode = match_mode.capitalize()
        self.cutoff = _utc(_date(include_all_before)) if include_all_before else None
        days = _days(start, end)
        if not days:
            print("Nothing to build: the span is empty.")
            return None
        all_days = [days[0] - dt.timedelta(days=i) for i in range(history_days, 0, -1)] + days
        # A ranked dataset needs the rank columns on its own days; warm-up days never do.
        needs = {d: DOWNLOAD_COLUMNS if match_mode == "Ranked" and d >= days[0] else BASE_COLUMNS
                 for d in all_days}
        self.folder.mkdir(parents=True, exist_ok=True)
        self.days_dir.mkdir(parents=True, exist_ok=True)

        if not self._estimate_and_ask(all_days, days, match_mode, needs, ask):
            return None
        self._download(all_days, needs)
        self._build(all_days, days, match_mode, history_days)
        return self.folder

    # ---------------------------------------------------------------- 1. estimate

    def _estimate_and_ask(self, all_days, days, match_mode, needs, ask):
        con = self.lake.con
        to_get = [d for d in all_days if not self._has(d, needs[d])]
        saved = [d for d in all_days if d not in to_get]
        print(f"Estimating the size ({len(saved)} of {len(all_days)} days already on your PC)...",
              flush=True)
        first_unix = int(_utc(days[0]).timestamp())

        # Days already saved: counted on your PC, which is fast and exact.
        rows = matches = 0
        if saved:
            files = "[" + ", ".join(text(self._day_file(d).as_posix()) for d in saved) + "]"
            rows, matches = con.execute(
                f"SELECT count(*), count(DISTINCT match_id) FILTER ("
                f"  WHERE {self._local_wanted(match_mode)} AND epoch(start_time) >= {first_unix}) "
                f"FROM read_parquet({files}, union_by_name = true)").fetchone()

        # Days still to download: count a few sample days on the server and scale up.
        # (Counting every day there would read the whole span and take many minutes.)
        to_get_rows, samples = 0, 0
        warm_up = [d for d in to_get if d < days[0]]
        span = [d for d in to_get if d >= days[0]]
        for group, in_span in ((warm_up, False), (span, True)):
            picked = _spread(group, 3)
            if not picked:
                continue
            counts = [self._count_day(d, match_mode) for d in picked]
            to_get_rows += sum(r for r, _ in counts) / len(picked) * len(group)
            if in_span:  # warm-up days only count experience, they add no matches
                matches += round(sum(m for _, m in counts) / len(picked) * len(group))
            samples += len(picked)
        if not matches:
            print("No matching matches found in that span.")
            return False

        # Bytes per downloaded player row, measured on a real sample.
        row_bytes = 0
        if to_get:
            sample = self.days_dir / "_sample.parquet"
            lo = _utc(to_get[0])
            for hours in (1, 6, 24):
                count = self._save(lo, lo + dt.timedelta(hours=hours), sample)
                if count >= 10_000:
                    break
            row_bytes = sample.stat().st_size / max(count, 1)
            sample.unlink()

        download = to_get_rows * row_bytes
        dataset = matches * BYTES_PER_MATCH
        free = shutil.disk_usage(self.folder).free
        print()
        kept = f"{match_mode.lower()} matches only"
        if self.cutoff:
            kept = f"all normal matches before {self.cutoff.date()}, then {kept}"
        print(f"  Dataset span:     {days[0]} to {days[-1]}, {kept}")
        print(f"  Warm-up history:  {len(all_days) - len(days)} days before that, "
              f"all normal matches, only to count experience")
        print(f"  Matches:          about {matches:,} (leavers and incomplete teams get dropped)")
        if to_get:
            print(f"  To download:      {_n_days(len(to_get))}, about {to_get_rows:,.0f} player rows"
                  f" x {row_bytes:,.0f} B = about {_size(download)}")
        else:
            print(f"  To download:      nothing, all {len(all_days)} days are already saved")
        if samples:
            print(f"                    (estimated from {samples} sample {_n_days(samples)[len(str(samples)) + 1:]}, so expect some error)")
        print(f"  Dataset:          {matches:,} x {BYTES_PER_MATCH} B = about {_size(dataset)}"
              f" (also its size in memory when training)")
        print(f"  Free disk:        {_size(free)} -> "
              f"{'enough' if free > (download + 2 * dataset) * 1.2 else 'NOT enough'}")
        print()
        if ask and input("Do you want to proceed? [Y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled, nothing downloaded.")
            return False
        return True

    def _count_day(self, day, match_mode):
        """(player rows, matches of the dataset's kind) on the server for one day."""
        wanted = f"{name('match_mode')} = {self.lake.values('match_mode', match_mode)}"
        if self.cutoff:
            wanted = f"({wanted} OR {name('start_time')} < {self.lake.time(self.cutoff)})"
        lo, hi = _utc(day), _utc(day + dt.timedelta(days=1))
        return self.lake.con.execute(
            f"SELECT count(*), count(DISTINCT match_id) FILTER (WHERE {wanted}) "
            f"FROM {self.lake.raw} WHERE {self._where(lo, hi)}").fetchone()

    def _local_wanted(self, match_mode):
        """SQL on saved day files: is this player row from a match the dataset keeps?"""
        wanted = f"match_mode = '{match_mode}'"
        if self.cutoff:  # before the cutoff every mode counts
            wanted = f"({wanted} OR epoch(start_time) < {int(self.cutoff.timestamp())})"
        return wanted

    # ---------------------------------------------------------------- 2. download

    def _download(self, days, needs):
        todo = [d for d in days if not self._has(d, needs[d])]
        print(f"Downloading {_n_days(len(todo))} ({len(days) - len(todo)} of {len(days)} already saved)."
              f" You can stop at any time; a re-run continues where it stopped.", flush=True)
        started = time.time()
        for i, day in enumerate(todo, 1):
            # The first half of the line shows straight away, the rest when the day is done.
            print(f"[{i}/{len(todo)}] {day}: downloading... ", end="", flush=True)
            day_start = time.time()
            count = self._save(_utc(day), _utc(day + dt.timedelta(days=1)), self._day_file(day))
            left = (time.time() - started) / i * (len(todo) - i)
            print(f"{count:,} player rows in {_duration(time.time() - day_start)}"
                  + (f", about {_duration(left)} left" if i < len(todo) else ""), flush=True)
        if todo:
            print(f"Download finished in {_duration(time.time() - started)}.", flush=True)

    def _has(self, day, columns):
        """Is this day saved, with every column this build needs? (Files from an
        older version may lack the rank columns; those days get downloaded again.)"""
        path = self._day_file(day)
        if not path.exists():
            return False
        saved = {r[0] for r in self.lake.con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({text(path.as_posix())})").fetchall()}
        return set(columns) <= saved

    def _day_file(self, day):
        return self.days_dir / f"{day}.parquet"

    def _save(self, lo, hi, path):
        """Player rows of normal ranked/unranked matches started in [lo, hi) -> a Parquet file."""
        columns = ", ".join(f"{self.lake.read(c)} AS {name(c)}" for c in DOWNLOAD_COLUMNS)
        part = path.with_name(path.name + ".part")
        count = self.lake.con.execute(
            f"COPY (SELECT {columns} FROM {self.lake.source} WHERE {self._where(lo, hi)}) "
            f"TO {text(part.as_posix())} (FORMAT parquet, COMPRESSION zstd)"
        ).fetchone()[0]
        part.replace(path)
        return count

    def _where(self, lo, hi):
        values = self.lake.values
        return " AND ".join([
            f"{name('start_time')} >= {self.lake.time(lo)}",
            f"{name('start_time')} < {self.lake.time(hi)}",
            f"{name('game_mode')} = {values('game_mode', 'Normal')}",
            f"{name('match_mode')} IN ({values('match_mode', ['Ranked', 'Unranked'])})",
            f"{name('match_outcome')} = {values('match_outcome', 'TeamWin')}",
            f"{name('team')} IN ({values('team', ['Team0', 'Team1'])})",
        ])

    # ---------------------------------------------------------------- 3. build

    def _build(self, all_days, days, match_mode, history_days):
        con = self.lake.con
        # info.json is written last, so it marks a finished dataset. Remove it first:
        # if this build stops halfway, the folder won't look finished.
        (self.folder / "info.json").unlink(missing_ok=True)
        files = "[" + ", ".join(text(self._day_file(d).as_posix()) for d in all_days) + "]"
        saved = {r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({files}, union_by_name = true)").fetchall()}

        lanes = [r[0] for r in con.execute(
            f"SELECT DISTINCT assigned_lane FROM read_parquet({files}, union_by_name = true) "
            f"WHERE assigned_lane IS NOT NULL ORDER BY 1").fetchall()]
        if any(lane > 15 for lane in lanes):
            print(f"Warning: unexpected lane numbers {lanes}; lanes above 15 are stored as 0.")

        print("Building step 1 of 2: counting experience and making one row per match. This runs"
              " on your PC with no progress lines and can take 10-30 minutes for many months...",
              flush=True)
        started = time.time()
        matches_file = self.folder / "matches.parquet"
        part = matches_file.with_name(matches_file.name + ".part")
        query = self._matches_query(files, saved, _utc(days[0]), _utc(days[-1] + dt.timedelta(days=1)),
                                    match_mode)
        con.execute(f"COPY ({query}) TO {text(part.as_posix())} (FORMAT parquet, COMPRESSION zstd)")
        part.replace(matches_file)
        print(f"  done in {_duration(time.time() - started)}.", flush=True)

        # Copy into .npy files, one day at a time so memory stays small.
        print(f"Building step 2 of 2: saving {len(days)} days as .npy files...", flush=True)
        source = f"read_parquet({text(matches_file.as_posix())})"
        n = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        save = np.lib.format.open_memmap
        arrays = {
            "inputs": save(self.folder / "inputs.npy", mode="w+", dtype=np.int16, shape=(n, len(INPUT_NAMES))),
            "outputs": save(self.folder / "outputs.npy", mode="w+", dtype=np.int16, shape=(n, len(OUTPUT_NAMES))),
            "rank_tiers": save(self.folder / "rank_tiers.npy", mode="w+", dtype=np.int8, shape=(n,)),
            "rank_scores": save(self.folder / "rank_scores.npy", mode="w+", dtype=np.float32, shape=(n,)),
            "match_ids": save(self.folder / "match_ids.npy", mode="w+", dtype=np.int64, shape=(n,)),
            "start_times": save(self.folder / "start_times.npy", mode="w+", dtype=np.int64, shape=(n,)),
        }
        row = 0
        for day in days:
            lo, hi = int(_utc(day).timestamp()), int(_utc(day + dt.timedelta(days=1)).timestamp())
            cols = con.execute(f"SELECT * FROM {source} WHERE start_unix >= {lo} AND start_unix < {hi} "
                               f"ORDER BY start_unix, match_id").fetchnumpy()
            k = len(cols["match_id"])
            arrays["inputs"][row:row + k] = _int16(np.column_stack([cols[c] for c in INPUT_NAMES]))
            arrays["outputs"][row:row + k] = _int16(np.column_stack([cols[c] for c in OUTPUT_NAMES]))
            arrays["rank_tiers"][row:row + k] = cols["rank_tier"]
            arrays["rank_scores"][row:row + k] = np.ma.filled(cols["rank_score"].astype(np.float32), np.nan)
            arrays["match_ids"][row:row + k] = cols["match_id"]
            arrays["start_times"][row:row + k] = cols["start_unix"]
            row += k
        tiers = np.array(arrays["rank_tiers"])
        for array in arrays.values():
            array.flush()
        del arrays

        info = {
            "version": DATASET_VERSION,
            "match_mode": match_mode,
            "include_all_before": str(self.cutoff.date()) if self.cutoff else None,
            "input_names": INPUT_NAMES,
            "output_names": OUTPUT_NAMES,
            "rank_names": RANK_NAMES,
            "lanes_seen": lanes,
            "start": str(days[0]),
            "end": str(days[-1]),
            "history_days": history_days,
            "matches": n,
        }
        (self.folder / "info.json").write_text(json.dumps(info, indent=2))
        size = sum(p.stat().st_size for p in self.folder.glob("*.npy"))
        print(f"Done: {n:,} matches saved in {self.folder} ({_size(size)}).")
        print("Matches per rank:", ", ".join(
            f"{name_}: {int((tiers == i).sum()):,}" for i, name_ in enumerate(["unknown"] + RANK_NAMES)
            if (tiers == i).any()))

    def _matches_query(self, files, saved, lo, hi, match_mode):
        """Player rows -> experience -> one row per match of this mode."""
        team0, team1 = "team = 'Team0'", "team = 'Team1'"
        # Files from an older version may lack some columns: read those as empty.
        players = ", ".join(name(c) if c in saved else f"CAST(NULL AS INTEGER) AS {name(c)}"
                            for c in DOWNLOAD_COLUMNS)

        per_player = []
        for col, _ in PER_PLAYER:
            for side, cond in ((0, team0), (1, team1)):
                per_player.append(f"list({col} ORDER BY player_slot) FILTER (WHERE {cond}) AS {col}_{side}")
        inputs = [f"coalesce({col}_{side}[{p}], 0) AS {prefix}_t{side}p{p}"
                  for col, prefix in PER_PLAYER for side in (0, 1) for p in range(1, 7)]
        claimed = name("mid_boss.team_claimed")
        wanted = self._local_wanted(match_mode)

        return f"""
        WITH players AS (
            SELECT {players} FROM read_parquet({files}, union_by_name = true)
        ),
        experience AS (   -- games and wins on this hero BEFORE each match (the row itself excluded)
            SELECT *,
                CASE WHEN account_id = 0 THEN 0 ELSE count(*) OVER earlier END AS games_before,
                CASE WHEN account_id = 0 THEN 0
                     ELSE coalesce(sum(CAST(won AS INTEGER)) OVER earlier, 0) END AS wins_before
            FROM players
            WINDOW earlier AS (PARTITION BY account_id, hero_id ORDER BY start_time, match_id
                               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ),
        chosen AS (
            SELECT *,
                CASE WHEN assigned_lane BETWEEN 1 AND 15 THEN assigned_lane ELSE 0 END AS lane,
                CASE WHEN coalesce(party, 0) = 0 THEN 1
                     ELSE count(*) OVER (PARTITION BY match_id, team, party) END AS party_size
            FROM experience
            WHERE {wanted}
              AND epoch(start_time) >= {int(lo.timestamp())} AND epoch(start_time) < {int(hi.timestamp())}
        ),
        matches AS (
            SELECT match_id,
                any_value(start_time) AS start_time,
                {", ".join(per_player)},
                any_value(winning_team) = 'Team0' AS team0_won,
                any_value(duration_s) AS duration_s,
                sum(kills) FILTER (WHERE {team0}) AS kills_team0,
                sum(kills) FILTER (WHERE {team1}) AS kills_team1,
                any_value({claimed}) AS claimed,
                any_value(coalesce(new_player_pool, false)) AS new_player_pool,
                any_value(coalesce(low_pri_pool, false)) AS low_pri_pool,
                any_value(coalesce(is_high_skill_range_parties, false)) AS high_skill_range_parties,
                avg({_rank_step('player_rank_initial_display_rank')}) AS players_rank,
                any_value({_rank_step('average_badge_team0')}) AS team0_rank,
                any_value({_rank_step('average_badge_team1')}) AS team1_rank,
                count(*) FILTER (WHERE {team0}) AS players_team0,
                count(*) FILTER (WHERE {team1}) AS players_team1,
                max(abandon_match_time_s) AS left_at
            FROM chosen
            GROUP BY match_id
        ),
        ranked AS (
            SELECT *,
                coalesce(players_rank,
                         (coalesce(team0_rank, team1_rank) + coalesce(team1_rank, team0_rank)) / 2.0)
                    AS rank_score
            FROM matches
            WHERE players_team0 = 6 AND players_team1 = 6 AND coalesce(left_at, 0) = 0
        )
        SELECT match_id,
            CAST(epoch(start_time) AS BIGINT) AS start_unix,
            {", ".join(inputs)},
            CAST(new_player_pool AS INTEGER) AS new_player_pool,
            CAST(low_pri_pool AS INTEGER) AS low_pri_pool,
            CAST(high_skill_range_parties AS INTEGER) AS high_skill_range_parties,
            CAST(team0_won AS INTEGER) AS team0_won,
            coalesce(duration_s, 0) AS duration_s,
            coalesce(kills_team0, 0) AS kills_team0,
            coalesce(kills_team1, 0) AS kills_team1,
            coalesce(len(list_filter(claimed, lambda x: x = 'Team0')), 0) AS midbosses_team0,
            coalesce(len(list_filter(claimed, lambda x: x = 'Team1')), 0) AS midbosses_team1,
            rank_score,
            CASE WHEN rank_score IS NULL THEN 0
                 ELSE (CAST(floor(rank_score + 0.5) AS INTEGER) - 1) // 6 + 1 END AS rank_tier
        FROM ranked
        ORDER BY start_unix, match_id
        """


def _rank_step(col):
    """A badge (tier * 10 + subrank, e.g. 82 = Oracle 2) as a step from 1 to 66, so
    ranks can be averaged: Initiate 1 = 1, Initiate 6 = 6, Seeker 1 = 7 ... Eternus 6 = 66.
    Anything else (0 = not calibrated, missing) becomes NULL and is left out."""
    return (f"CASE WHEN {col} BETWEEN 11 AND 116 AND {col} % 10 BETWEEN 1 AND 6 "
            f"THEN ({col} // 10 - 1) * 6 + {col} % 10 END")


def _days(start, end):
    """Whole UTC days from start up to (not including) end; never today, which isn't over."""
    today = dt.datetime.now(dt.timezone.utc).date()
    first = _date(start)
    last = min(_date(end) if end else today, today) - dt.timedelta(days=1)
    if end and _date(end) > today:
        print(f"Note: stopping at {last}, the last full day.")
    return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]


def _date(value):
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    return value.date() if isinstance(value, dt.datetime) else value


def _utc(day):
    return dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)


def _int16(values):
    return np.clip(values, -32768, 32767).astype(np.int16)


def _n_days(n):
    return f"{n} day" if n == 1 else f"{n} days"


def _spread(items, k):
    """Up to k items spread evenly from first to last."""
    if len(items) <= k:
        return list(items)
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _duration(seconds):
    """7 -> '7s', 125 -> '2m 5s', 10_000 -> '2h 46m'."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {seconds % 3600 // 60}m"


def _size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"
