"""Regression guard for the row-insertion-timestamp wrapper (``pose_pipeline/dj_schema.py``).

Encodes the lesson from the insertion_time incident: a *uniform* secondary-attribute name on every
table (``insertion_time``) breaks DataJoint natural joins, because DataJoint refuses to join two
expressions that share a dependent (non-primary-key) attribute. The fix is a *distinct*
``<table>_inserted_at`` per table. These tests fail if anyone regresses the naming back to a shared
name, drops the 64-char safety, or loses idempotency -- and (with a DB) that stamped tables still join.
"""
import datajoint as dj
import pytest

from pose_pipeline.dj_schema import (
    _MAX_ATTR_LEN,
    TimestampedSchema,
    add_inserted_at,
    inserted_at_attr,
)


def _table(name, base=dj.Computed, definition="x : int"):
    """Build a throwaway table class (not declared against any schema)."""
    return type(name, (base,), {"definition": definition})


# --- naming rule (no DB needed) ------------------------------------------------------------------

def test_name_is_readable_per_table():
    assert inserted_at_attr(_table("Video")) == "video_inserted_at"
    assert inserted_at_attr(_table("KinematicReconstruction")) == "kinematic_reconstruction_inserted_at"


def test_names_are_distinct_across_tables():
    # THE core invariant: two different tables must never get the same stamped attribute name,
    # otherwise `A * B` collides on a shared secondary attribute (the original breakage).
    classes = ["Video", "VideoInfo", "TopDownPerson", "BottomUpPeople", "SMPLPerson", "PersonBbox"]
    names = [inserted_at_attr(_table(c)) for c in classes]
    assert len(names) == len(set(names)), f"stamped names collide: {names}"


def test_every_name_ends_in_the_greppable_suffix():
    for c in ["Video", "A" * 80]:
        assert inserted_at_attr(_table(c)).endswith("_inserted_at")


def test_add_is_idempotent():
    cls = _table("Video")
    add_inserted_at(cls)
    once = cls.definition
    add_inserted_at(cls)
    assert cls.definition == once
    assert cls.definition.count("video_inserted_at") == 1


def test_long_name_falls_back_within_limit_and_stays_unique():
    a, b = _table("A" * 80), _table("A" * 79)  # both would exceed 64 as <name>_inserted_at
    for c in (a, b):
        assert len(inserted_at_attr(c)) <= _MAX_ATTR_LEN
    assert inserted_at_attr(a) != inserted_at_attr(b)  # hash disambiguates truncated names


def test_lookup_and_part_are_not_stamped():
    assert "inserted_at" not in add_inserted_at(_table("Cfg", dj.Lookup)).definition
    assert "inserted_at" not in add_inserted_at(_table("Part", dj.Part)).definition


# --- integration with the live schema (needs a DataJoint DB) -------------------------------------

@pytest.fixture(scope="module")
def _db():
    try:
        dj.conn()
    except Exception as e:  # no DB configured (e.g. plain unit CI) -> skip the integration checks
        pytest.skip(f"no DataJoint connection: {e}")


def test_wrapper_is_used_and_no_uniform_name_remains(_db):
    import pose_pipeline.pipeline as P

    stamped = [
        t
        for t in vars(P).values()
        if isinstance(t, type)
        and issubclass(t, (dj.Computed, dj.Imported, dj.Manual))
        and not issubclass(t, dj.Part)
        and getattr(t, "database", None) == P.schema.database
    ]
    assert stamped, "expected pose_pipeline tables to introspect"
    for t in stamped:
        names = t.heading.names
        assert "insertion_time" not in names, f"{t.__name__} still has the uniform insertion_time"
        assert inserted_at_attr(t) in names, f"{t.__name__} missing {inserted_at_attr(t)}"


def test_representative_join_builds(_db):
    # the exact operation that failed under the uniform name must build without raising.
    from pose_pipeline.pipeline import TopDownPerson, Video, VideoInfo

    (Video * VideoInfo).heading  # noqa: B018  -- raised "Cannot join ... dependent attribute" before
    (Video * TopDownPerson).heading  # noqa: B018
