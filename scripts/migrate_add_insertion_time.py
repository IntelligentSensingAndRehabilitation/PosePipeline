"""One-time migration to add `insertion_time` to existing pose_pipeline tables.

Pairs with the `TimestampedSchema` change (pose_pipeline/dj_schema.py), which stamps *new*
tables automatically. This migration updates the *existing* database:

  - ADD    a `insertion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP` column to every
           Computed/Imported/Manual table that doesn't have one (Lookups/Parts skipped), using
           ALGORITHM=INSTANT (metadata-only, no table rebuild -- fast even on the largest tables).
           Existing rows get the migration time; new rows get their insert time.
  - RENAME the 3 legacy timestamp fields to `insertion_time` via RENAME COLUMN (a pure metadata
           rename that preserves the *real* historical timestamps): Video.import_time,
           BottomUpPeople.timestamp, SkeletonAction.computed_timestamp.

Locking: ALTERs run one table at a time (one brief metadata lock each), so this does not lock the
whole DB. Each op is metadata-only/instant; the only stall risk is if a long-running transaction
holds a lock on the specific table being altered -- prefer a quiet window. A per-table failure is
reported and the run continues (idempotent: re-run to retry).

Idempotent (safe to re-run). DRY-RUN by default; pass --apply to execute. Requires WRITE creds.

    python scripts/migrate_add_insertion_time.py            # dry-run: show what would change
    python scripts/migrate_add_insertion_time.py --apply    # execute
"""
import argparse

import datajoint as dj

# tables whose existing timestamp field is RENAMED (keeps real historical values, not migration time)
RENAMES = {
    "Video": "import_time",
    "BottomUpPeople": "timestamp",
    "SkeletonAction": "computed_timestamp",
}
COLDEF = "`insertion_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="execute the ALTERs (default: dry-run)")
    args = ap.parse_args()

    import pose_pipeline.pipeline as P

    conn = dj.conn()
    db = P.schema.database

    tables = [
        obj
        for obj in vars(P).values()
        if isinstance(obj, type)
        and issubclass(obj, (dj.Computed, dj.Imported, dj.Manual))
        and not issubclass(obj, dj.Part)
        and getattr(obj, "database", None) == db
    ]
    print(f"schema: {db} | {len(tables)} Computed/Imported/Manual tables\n")

    add = rename = skip = fail = 0
    for t in sorted(tables, key=lambda c: c.__name__):
        names = t.heading.names
        ftn = t.full_table_name
        if "insertion_time" in names:
            print(f"  skip    {t.__name__:30s} (already has insertion_time)")
            skip += 1
            continue
        legacy = RENAMES.get(t.__name__)
        if legacy and legacy in names:
            # pure metadata rename: preserves the column's type/default and its real historical values
            sql = f"ALTER TABLE {ftn} RENAME COLUMN `{legacy}` TO `insertion_time`"
            print(f"  RENAME  {t.__name__:30s} {legacy} -> insertion_time")
            rename += 1
        else:
            # ALGORITHM=INSTANT: metadata-only add (no table rebuild); errors loudly if not possible
            sql = f"ALTER TABLE {ftn} ADD COLUMN {COLDEF}, ALGORITHM=INSTANT"
            print(f"  ADD     {t.__name__:30s}")
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
