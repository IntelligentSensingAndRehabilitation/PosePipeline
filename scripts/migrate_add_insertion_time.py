"""One-time migration: give every pose_pipeline table a `<table>_inserted_at` timestamp column.

Pairs with the `TimestampedSchema` wrapper (pose_pipeline/dj_schema.py), which stamps *new* tables
automatically; this brings the *existing* database in line. Per table (Lookups/Parts skipped), the
target name is `inserted_at_attr(table)` (`<table>_inserted_at`), and:

  - skip   if the table already has it;
  - RENAME `insertion_time` -> target, if present (this repo's earlier rollout used the uniform
           `insertion_time` name, which broke DataJoint joins -- two tables can't share a secondary
           attribute; the per-table name fixes that). Pure metadata rename, keeps historical values;
  - RENAME a legacy timestamp field -> target (see RENAMES), converting datetime->timestamp if needed;
  - ADD    the column via ALGORITHM=INSTANT (metadata-only) otherwise. Existing rows get the
           migration time; new rows get their insert time.

Locking: ALTERs run one table at a time (one brief metadata lock each), so this does not lock the
whole DB. A per-table failure is reported and the run continues.

Idempotent (safe to re-run). DRY-RUN by default; pass --apply to execute. Requires WRITE creds.

    python scripts/migrate_add_insertion_time.py            # dry-run: show what would change
    python scripts/migrate_add_insertion_time.py --apply    # execute
"""
import argparse
import importlib

import datajoint as dj

from pose_pipeline.dj_schema import inserted_at_attr

# modules to import so their table classes are in scope
MODULES = ["pose_pipeline.pipeline"]
# only migrate tables that BELONG to these schemas (modules may import tables from other schemas)
SCHEMAS = {"pose_pipeline"}
# tables whose pre-existing (non-insertion_time) legacy timestamp field should be renamed instead of
# adding a new column (ClassName -> legacy field). None here: this repo already ran the earlier
# rollout, so its tables now carry `insertion_time`, which the RENAME-insertion_time branch handles.
RENAMES = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute the ALTERs (default: dry-run)")
    args = ap.parse_args()

    conn = dj.conn()
    tables, seen = [], set()
    for mn in MODULES:
        mod = importlib.import_module(mn)
        for obj in vars(mod).values():
            if (
                isinstance(obj, type)
                and issubclass(obj, (dj.Computed, dj.Imported, dj.Manual))
                and not issubclass(obj, dj.Part)
                and getattr(obj, "database", None) in SCHEMAS  # skip tables imported from other schemas
            ):
                ftn = getattr(obj, "full_table_name", None)
                if ftn and ftn not in seen:
                    seen.add(ftn)
                    tables.append(obj)
    print(f"{len(tables)} Computed/Imported/Manual tables across {len(MODULES)} module(s)\n")

    add = rename = skip = fail = 0
    for t in sorted(tables, key=lambda c: c.__name__):
        names = t.heading.names
        ftn = t.full_table_name
        target = inserted_at_attr(t)
        coldef = f"`{target}` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"
        if target in names:
            print(f"  skip    {t.__name__:34s} (has {target})")
            skip += 1
            continue
        if "insertion_time" in names:  # correct the earlier uniform-name rollout
            sql = f"ALTER TABLE {ftn} RENAME COLUMN `insertion_time` TO `{target}`"
            print(f"  RENAME  {t.__name__:34s} insertion_time -> {target}")
            rename += 1
        elif (legacy := RENAMES.get(t.__name__)) and legacy in names:
            typ = str(t.heading.attributes[legacy].type)
            if typ.lower().startswith("timestamp"):
                sql = f"ALTER TABLE {ftn} RENAME COLUMN `{legacy}` TO `{target}`"
            else:  # legacy `datetime` field -> convert to timestamp while renaming
                sql = f"ALTER TABLE {ftn} CHANGE COLUMN `{legacy}` {coldef}"
            print(f"  RENAME  {t.__name__:34s} {legacy} ({typ}) -> {target}")
            rename += 1
        else:
            sql = f"ALTER TABLE {ftn} ADD COLUMN {coldef}, ALGORITHM=INSTANT"
            print(f"  ADD     {t.__name__:34s} {target}")
            add += 1
        if args.apply:
            try:
                conn.query(sql)
            except Exception as e:  # keep going so one table can't halt the run; report at the end
                print(f"          FAILED ({type(e).__name__}: {str(e)[:100]})")
                fail += 1

    print(f"\n{add} ADD, {rename} RENAME, {skip} already done" + (f", {fail} FAILED" if fail else "") + ".")
    if args.apply:
        print("APPLIED with FAILURES — re-run after investigating." if fail else "APPLIED ✅")
    else:
        print("DRY-RUN — pass --apply to execute.")


if __name__ == "__main__":
    main()
