"""One-time migration to add `insertion_time` to existing pose_pipeline tables.

Pairs with the `TimestampedSchema` change (pose_pipeline/dj_schema.py), which stamps *new*
tables automatically. This migration updates the *existing* database:

  - ADD    a `insertion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP` column to every
           Computed/Imported/Manual table that doesn't have one (Lookups/Parts skipped).
           Existing rows get the migration time; new rows get their insert time.
  - RENAME the 3 legacy timestamp fields to `insertion_time` (preserving the *real* historical
           timestamps those tables already recorded): Video.import_time,
           BottomUpPeople.timestamp, SkeletonAction.computed_timestamp.

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

    add = rename = skip = 0
    for t in sorted(tables, key=lambda c: c.__name__):
        names = t.heading.names
        ftn = t.full_table_name
        if "insertion_time" in names:
            print(f"  skip    {t.__name__:30s} (already has insertion_time)")
            skip += 1
            continue
        legacy = RENAMES.get(t.__name__)
        if legacy and legacy in names:
            sql = f"ALTER TABLE {ftn} CHANGE COLUMN `{legacy}` {COLDEF}"
            print(f"  RENAME  {t.__name__:30s} {legacy} -> insertion_time")
            rename += 1
        else:
            sql = f"ALTER TABLE {ftn} ADD COLUMN {COLDEF}"
            print(f"  ADD     {t.__name__:30s}")
            add += 1
        if args.apply:
            conn.query(sql)

    print(f"\n{add} ADD, {rename} RENAME, {skip} already done.")
    print("APPLIED ✅" if args.apply else "DRY-RUN — pass --apply to execute.")


if __name__ == "__main__":
    main()
