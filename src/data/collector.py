"""
collector.py - collect Deadlock matches, after telling you how big the download will be.

    from deadlock_data import MatchCollector

    collector = MatchCollector()
    collector.columns()                   # print every column you can ask for

    collector.collect(
        columns=["duration_s", "hero_id"],
        start="2026-09-29", end="2026-10-12",
        match_mode="Unranked", game_mode="Normal",
        out_dir="data/unranked",
    )
    df = MatchCollector.load("data/unranked")

Days are whole UTC days and `end` is not included: end="2026-10-12" stops at the
end of Oct 11.
"""

import datetime as dt
import shutil
from pathlib import Path

from .lake import Lake, name, read_folder, text


class MatchCollector:
    def __init__(self, cache_dir="deadlock_cache", **lake_options):
        print("Connecting to the data lake...")
        self.lake = Lake(cache_dir, **lake_options)

    def columns(self):
        """Print every column: name, match/player level, type, description."""
        for col, col_type in self.lake.types.items():
            print(f"{col:46} {self.lake.level(col):7} {col_type:26} {self.lake.comments.get(col, '')}")

    @staticmethod
    def load(out_dir):
        """Read a folder filled by collect() into one pandas DataFrame."""
        return read_folder(out_dir)

    def collect(self, columns, start, end=None, *, match_mode=("Ranked", "Unranked"),
                game_mode="Normal", out_dir="data/matches", team_size=6, ask=True):
        """Count, estimate the size, ask, then download one file per day into out_dir.

        Every row is one match: match_id, start_time, team0_won, plus your columns.
        A player column such as hero_id becomes two lists, hero_id_team0 and
        hero_id_team1, ordered by player slot.
        match_mode, game_mode: one value, a list, or None for no filter.
        team_size: keep only matches with this many players per side (None = any).
        """
        columns = self._check(columns)
        days = self._days(start, end)
        if not days:
            print("Nothing to collect: the span is empty.")
            return None
        lo, hi = _utc(days[0]), _utc(days[-1] + dt.timedelta(days=1))
        where = self._filters(match_mode, game_mode)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. How many matches? (reads only a few small columns)
        print(f"Counting matches from {days[0]} to {days[-1]} (UTC)...")
        matches = self.lake.con.execute(
            f"SELECT count(DISTINCT match_id) FROM {self.lake.raw} "
            f"WHERE {self._between(lo, hi)} AND {where}"
        ).fetchone()[0]
        if not matches:
            print("No matches found for these filters.")
            return None

        # 2. How big is one row? Save a small real sample and measure it,
        #    both as a file on disk and loaded into memory (RAM).
        row_disk, row_ram, sampled = self._measure_row(columns, lo, hi, where, team_size, out)

        # 3. Totals, and disk space check.
        total = matches * row_disk
        free = shutil.disk_usage(out).free
        print()
        print(f"  Matches found:  about {matches:,}")
        print(f"  One row:        about {row_disk:,.0f} bytes on disk, {row_ram:,.0f} bytes in memory"
              f" (measured on {sampled:,} matches)")
        print(f"  On disk:        {matches:,} x {row_disk:,.0f} B = about {_size(total)}")
        print(f"  In memory:      {matches:,} x {row_ram:,.0f} B = about {_size(matches * row_ram)}"
              f" when loaded with MatchCollector.load()")
        print(f"  Free disk:      {_size(free)} -> {'enough' if free > total * 1.2 else 'NOT enough'}")
        print()
        if ask and input("Do you want to proceed? [Y/N] ").strip().lower() not in ("y", "yes"):
            print("Cancelled, nothing downloaded.")
            return None

        # 4. Download, one file per day. Days already saved are skipped, so you can
        #    stop the program and run it again.
        self._remember_query(out, columns, match_mode, game_mode, team_size)
        for i, day in enumerate(days, 1):
            path = out / f"{day}.parquet"
            if path.exists():
                print(f"[{i}/{len(days)}] {day}: already saved")
                continue
            count = self._save(columns, _utc(day), _utc(day + dt.timedelta(days=1)), where, team_size, path)
            print(f"[{i}/{len(days)}] {day}: {count:,} matches")
        print(f"Done. Folder size: {_size(sum(p.stat().st_size for p in out.glob('*.parquet')))}")
        return out

    # ---------------------------------------------------------------- the steps

    def _measure_row(self, columns, lo, hi, where, team_size, out):
        """Bytes per match row on disk and in memory, from a real sample saved the same
        way as the download. Tries a bigger time window until it has 1,000 matches."""
        sample = out / "_sample.parquet"
        for hours in (1, 6, 24, 24 * 7):
            window_end = min(lo + dt.timedelta(hours=hours), hi)
            count = self._save(columns, lo, window_end, where, team_size, sample)
            if count >= 1000 or window_end == hi:
                break
        disk = sample.stat().st_size
        loaded = self.lake.con.execute(f"SELECT * FROM read_parquet({text(sample.as_posix())})").df()
        ram = loaded.memory_usage(deep=True).sum()  # what pandas really uses for these rows
        sample.unlink()
        return disk / max(count, 1), ram / max(count, 1), count

    def _save(self, columns, lo, hi, where, team_size, path):
        """Run the query for [lo, hi) and write the result straight to a Parquet file."""
        part = path.with_name(path.name + ".part")
        count = self.lake.con.execute(
            f"COPY ({self._query(columns, lo, hi, where, team_size)}) "
            f"TO {text(part.as_posix())} (FORMAT parquet, COMPRESSION zstd)"
        ).fetchone()[0]
        part.replace(path)  # only a finished file gets its real name
        return count

    def _query(self, columns, lo, hi, where, team_size):
        """Step 1: the player rows. Step 2: group them into one row per match."""
        read = self.lake.read
        inner = ["match_id", "start_time", "winning_team", "team", "player_slot"]
        inner += [c for c in columns if c not in inner]
        team0, team1 = f"{name('team')} = 'Team0'", f"{name('team')} = 'Team1'"

        outer = [
            name("match_id"),
            f"any_value({name('start_time')}) AS {name('start_time')}",
            f"any_value({name('winning_team')}) = 'Team0' AS team0_won",
        ]
        for col in columns:
            if col in ("match_id", "start_time"):
                continue
            if self.lake.level(col) == "match":
                outer.append(f"any_value({name(col)}) AS {name(col)}")
            else:
                for side, cond in (("team0", team0), ("team1", team1)):
                    outer.append(f"list({name(col)} ORDER BY {name('player_slot')}) "
                                 f"FILTER (WHERE {cond}) AS {name(col + '_' + side)}")

        having = ""
        if team_size:
            having = (f"HAVING count(*) FILTER (WHERE {team0}) = {int(team_size)} "
                      f"AND count(*) FILTER (WHERE {team1}) = {int(team_size)}")
        return (
            f"WITH players AS ("
            f"  SELECT {', '.join(f'{read(c)} AS {name(c)}' for c in inner)} FROM {self.lake.source}"
            f"  WHERE {self._between(lo, hi)} AND {where}"
            f"  AND {name('team')} IN ({self.lake.values('team', ['Team0', 'Team1'])})) "
            f"SELECT {', '.join(outer)} FROM players GROUP BY {name('match_id')} {having} "
            f"ORDER BY start_time, match_id"
        )

    # ---------------------------------------------------------------- small helpers

    def _filters(self, match_mode, game_mode):
        filters = [f"{name('match_outcome')} = {self.lake.values('match_outcome', 'TeamWin')}"]
        if match_mode:
            filters.append(f"{name('match_mode')} IN ({self.lake.values('match_mode', match_mode)})")
        if game_mode:
            filters.append(f"{name('game_mode')} IN ({self.lake.values('game_mode', game_mode)})")
        return " AND ".join(filters)

    def _between(self, lo, hi):
        return f"{name('start_time')} >= {self.lake.time(lo)} AND {name('start_time')} < {self.lake.time(hi)}"

    def _check(self, columns):
        columns = [columns] if isinstance(columns, str) else list(dict.fromkeys(columns))
        for col in columns:
            if col not in self.lake.types:
                raise ValueError(f"Unknown column {col!r}. Run collector.columns() to see all names.")
        return columns

    def _days(self, start, end):
        """Whole UTC days from start up to (not including) end. Today is left out
        because it isn't over yet, so its file would be incomplete."""
        today = dt.datetime.now(dt.timezone.utc).date()
        first = _date(start)
        last = min(_date(end) if end else today, today) - dt.timedelta(days=1)
        if end and _date(end) > today:
            print(f"Note: stopping at {last}, the last full day.")
        return [first + dt.timedelta(days=i) for i in range((last - first).days + 1)]

    def _remember_query(self, out, columns, match_mode, game_mode, team_size):
        """A folder holds one kind of query, so files with different columns never mix."""
        query = repr((columns, match_mode, game_mode, team_size))
        note = out / "_query.txt"
        if note.exists() and note.read_text() != query:
            raise ValueError(f"{out} already holds a different query ({note.read_text()}). "
                             f"Use another out_dir or empty this one.")
        note.write_text(query)


def _date(value):
    if isinstance(value, str):
        return dt.date.fromisoformat(value[:10])
    return value.date() if isinstance(value, dt.datetime) else value


def _utc(day):
    return dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc)


def _size(n_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"
