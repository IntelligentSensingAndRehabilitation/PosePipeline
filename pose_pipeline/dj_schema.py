"""Schema helper that stamps every table with a row-insertion timestamp.

`TimestampedSchema` is a drop-in replacement for `dj.schema`: it behaves exactly like a normal
DataJoint schema, but when it declares a table it first appends an ``insertion_time`` secondary
attribute to the table's definition. This gives us a uniform, automatic record of when each row
was inserted (useful for provenance and for measuring processing time between pipeline stages),
without having to add the field by hand to every table -- and so that *new* tables can't forget it.

Usage (replaces ``schema = dj.schema("...")``)::

    from pose_pipeline.dj_schema import TimestampedSchema
    schema = TimestampedSchema(db_prefix + "pose_pipeline")

Scope: only ``dj.Computed`` / ``dj.Imported`` / ``dj.Manual`` tables are stamped. ``dj.Lookup``
tables (static config) and ``dj.Part`` tables (inserted with their master) are skipped. A table
that already declares ``insertion_time`` is left untouched.

For existing databases the column is added by the one-time migration
(``scripts/migrate_add_insertion_time.py``); this wrapper covers freshly-declared tables.
"""

import re

import datajoint as dj

# `= CURRENT_TIMESTAMP` -> MySQL `DEFAULT CURRENT_TIMESTAMP` (set once at insert, no ON UPDATE).
INSERTION_TIME_ATTR = "insertion_time = CURRENT_TIMESTAMP : timestamp  # row insertion time (auto)"

_DIVIDER = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)


def add_insertion_time(cls):
    """Append an ``insertion_time`` attribute to a table class's ``definition`` (in place).

    Returns the class unchanged if it is a Lookup/Part, is not a table, or already has the field.
    """
    if not (isinstance(cls, type) and issubclass(cls, (dj.Computed, dj.Imported, dj.Manual))):
        return cls  # Lookup tables and non-tables: skip
    if issubclass(cls, dj.Part):
        return cls  # Part tables ride along with their master
    definition = getattr(cls, "definition", None)
    if not isinstance(definition, str) or "insertion_time" in definition:
        return cls  # already stamped (e.g. the renamed legacy tables)

    body = definition.rstrip()
    if not _DIVIDER.search(body):
        body += "\n    ---"  # all-primary-key table: it has no secondary section yet
    cls.definition = body + f"\n    {INSERTION_TIME_ATTR}\n    "
    return cls


class TimestampedSchema(dj.Schema):
    """A ``dj.Schema`` that stamps every Computed/Imported/Manual table with ``insertion_time``."""

    def __call__(self, cls, *args, **kwargs):
        add_insertion_time(cls)
        return super().__call__(cls, *args, **kwargs)
