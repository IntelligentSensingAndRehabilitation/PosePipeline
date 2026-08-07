"""Schema helper that stamps every table with a row-insertion timestamp.

``TimestampedSchema`` is a drop-in replacement for ``dj.schema``: it behaves exactly like a normal
DataJoint schema, but when it declares a table it first appends a row-insertion timestamp attribute
to the table's definition. This gives us a uniform, automatic record of when each row was inserted
(useful for provenance and for measuring processing time between pipeline stages), without having to
add the field by hand to every table -- and so that *new* tables can't forget it.

Usage (replaces ``schema = dj.schema("...")``)::

    from pose_pipeline.dj_schema import TimestampedSchema
    schema = TimestampedSchema(db_prefix + "pose_pipeline")

Naming: the attribute is ``<table>_inserted_at`` where ``<table>`` is the table's own snake_case
name (e.g. ``video_inserted_at``, ``kinematic_reconstruction_inserted_at``). The per-table prefix is
deliberate: DataJoint refuses to natural-join two expressions that share a *secondary* attribute, so
a single uniform name like ``insertion_time`` on every table would break ``A * B`` joins between any
two stamped tables. A name unique to each table keeps joins working. (The only residual case is a
natural join of two *identically-named* tables from different schemas -- rare and unusual.)

Scope: only ``dj.Computed`` / ``dj.Imported`` / ``dj.Manual`` tables are stamped. ``dj.Lookup``
tables (static config) and ``dj.Part`` tables (inserted with their master) are skipped. A table that
already declares its ``<table>_inserted_at`` field is left untouched.

For existing databases the column is added/renamed by the one-time migration
(``scripts/migrate_add_insertion_time.py`` in each repo); this wrapper covers freshly-declared tables.
"""

import hashlib
import re

import datajoint as dj
from datajoint.utils import from_camel_case

# DataJoint/MySQL cap attribute names at 64 chars.
_MAX_ATTR_LEN = 64
_SUFFIX = "_inserted_at"
_HASH_LEN = 8  # hex chars of the disambiguating hash used only in the abbreviation fallback

_DIVIDER = re.compile(r"^\s*-{3,}\s*$", re.MULTILINE)


def inserted_at_attr(cls):
    """Return the row-insertion-timestamp attribute name for a table class.

    Normally ``<table>_inserted_at`` (readable). If that would exceed the 64-char attribute limit
    (only for pathologically long table names), fall back to ``<prefix>_<hash>_inserted_at``: a
    truncated readable prefix plus a short deterministic hash of the full name, so it stays unique
    (plain truncation could collide) and still ends in the greppable ``_inserted_at``. This is a
    pure function -- the migration imports it too, so the code-generated name and the DB column
    can never diverge.
    """
    base = from_camel_case(cls.__name__)
    field = f"{base}{_SUFFIX}"
    if len(field) <= _MAX_ATTR_LEN:
        return field
    digest = hashlib.sha1(base.encode()).hexdigest()[:_HASH_LEN]
    keep = _MAX_ATTR_LEN - len(_SUFFIX) - _HASH_LEN - 1  # 1 for the underscore before the hash
    prefix = base[:keep].rstrip("_")
    return f"{prefix}_{digest}{_SUFFIX}"


def add_inserted_at(cls):
    """Append a ``<table>_inserted_at`` attribute to a table class's ``definition`` (in place).

    Returns the class unchanged if it is a Lookup/Part, is not a table, or already has the field.
    """
    if not (isinstance(cls, type) and issubclass(cls, (dj.Computed, dj.Imported, dj.Manual))):
        return cls  # Lookup tables and non-tables: skip
    if issubclass(cls, dj.Part):
        return cls  # Part tables ride along with their master
    definition = getattr(cls, "definition", None)
    if not isinstance(definition, str):
        return cls

    field = inserted_at_attr(cls)
    assert len(field) <= _MAX_ATTR_LEN, field  # inserted_at_attr guarantees this
    if field in definition:
        return cls  # already stamped

    body = definition.rstrip()
    if not _DIVIDER.search(body):
        body += "\n    ---"  # all-primary-key table: it has no secondary section yet
    cls.definition = body + f"\n    {field} = CURRENT_TIMESTAMP : timestamp  # row insertion time (auto)\n    "
    return cls


class TimestampedSchema(dj.Schema):
    """A ``dj.Schema`` that stamps every Computed/Imported/Manual table with ``<table>_inserted_at``."""

    def __call__(self, cls, *args, **kwargs):
        add_inserted_at(cls)
        return super().__call__(cls, *args, **kwargs)
