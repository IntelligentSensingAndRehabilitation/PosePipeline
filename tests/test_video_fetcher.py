from pathlib import Path
import importlib.util

import pytest

try:
    from pose_pipeline.utils.video_fetcher import VideoFetcher, cleanup_video_files
except ModuleNotFoundError:
    MODULE_PATH = Path(__file__).resolve().parents[1] / "pose_pipeline" / "utils" / "video_fetcher.py"
    SPEC = importlib.util.spec_from_file_location("video_fetcher", MODULE_PATH)
    video_fetcher = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(video_fetcher)

    VideoFetcher = video_fetcher.VideoFetcher
    cleanup_video_files = video_fetcher.cleanup_video_files


class DummyQuery:
    """Test double for a DataJoint relation used as direct restrictions and fetch targets."""

    def __init__(self, path=None, count=1, field_paths=None, fetch_exc=None):
        self.path = path
        self.count = count
        self.field_paths = field_paths or {}
        self.fetch_exc = fetch_exc

    def __len__(self):
        return self.count

    def fetch1(self, field):
        if self.fetch_exc is not None:
            raise self.fetch_exc
        if field in self.field_paths:
            return str(self.field_paths[field])
        if field == "video" and self.path is not None:
            return str(self.path)
        raise ValueError("Unexpected field")


class DummyVideoTable:
    """Test double for the DataJoint Video table restriction operator."""

    def __init__(self, result_by_restriction):
        self.result_by_restriction = result_by_restriction

    def __and__(self, restriction):
        if isinstance(restriction, DummyQuery):
            return restriction
        return self.result_by_restriction[repr(restriction)]


def test_cleanup_video_files_removes_files(tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("data")

    cleanup_video_files([video_file])

    assert not video_file.exists()


def test_video_fetcher_fetch_video_with_dict_restriction(tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("data")
    restriction = {"video_id": 1}
    table = DummyVideoTable({repr(restriction): DummyQuery(video_file, count=1)})

    with VideoFetcher(video_table=table) as fetcher:
        fetched = fetcher.fetch_video(restriction)
        assert fetched == video_file
        assert fetched.exists()

    assert not video_file.exists()


def test_video_fetcher_fetch_video_with_query_restriction(tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("data")
    query = DummyQuery(video_file, count=1)

    with VideoFetcher(video_table=DummyVideoTable({})) as fetcher:
        fetched = fetcher.fetch_video(query)
        assert fetched == video_file

    assert not video_file.exists()


def test_video_fetcher_fetch_videos_batch(tmp_path):
    video_file_1 = tmp_path / "video1.mp4"
    video_file_2 = tmp_path / "video2.mp4"
    video_file_1.write_text("data")
    video_file_2.write_text("data")
    key_1 = {"video_id": 1}
    key_2 = {"video_id": 2}
    table = DummyVideoTable(
        {
            repr(key_1): DummyQuery(video_file_1, count=1),
            repr(key_2): DummyQuery(video_file_2, count=1),
        }
    )

    with VideoFetcher(video_table=table) as fetcher:
        fetched = fetcher.fetch_videos([key_1, key_2])
        assert fetched == [video_file_1, video_file_2]

    assert not video_file_1.exists()
    assert not video_file_2.exists()


def test_video_fetcher_raises_when_not_exactly_one_match(tmp_path):
    video_file = tmp_path / "video.mp4"
    restriction = {"video_id": 1}
    table = DummyVideoTable({repr(restriction): DummyQuery(video_file, count=0)})

    with VideoFetcher(video_table=table) as fetcher:
        with pytest.raises(ValueError, match="must match exactly 1 video"):
            fetcher.fetch_video(restriction)


def test_video_fetcher_cleans_up_when_exception_is_raised(tmp_path):
    video_file = tmp_path / "video.mp4"
    video_file.write_text("data")
    restriction = {"video_id": 1}
    table = DummyVideoTable({repr(restriction): DummyQuery(video_file, count=1)})

    with pytest.raises(RuntimeError):
        with VideoFetcher(video_table=table) as fetcher:
            fetcher.fetch_video(restriction)
            raise RuntimeError("boom")

    assert not video_file.exists()


def test_video_fetcher_fetch_videos_partial_failure_cleans_previous_file(tmp_path):
    first_video = tmp_path / "video1.mp4"
    first_video.write_text("data")
    key_1 = {"video_id": 1}
    key_2 = {"video_id": 2}
    table = DummyVideoTable(
        {
            repr(key_1): DummyQuery(first_video, count=1),
            repr(key_2): DummyQuery(count=0),
        }
    )

    with pytest.raises(ValueError, match="must match exactly 1 video"):
        with VideoFetcher(video_table=table) as fetcher:
            fetcher.fetch_videos([key_1, key_2])

    assert not first_video.exists()


def test_video_fetcher_fetch_video_with_custom_attachment_field(tmp_path):
    output_video = tmp_path / "output_video.mp4"
    output_video.write_text("data")
    restriction = {"video_id": 1}
    table = DummyVideoTable(
        {
            repr(restriction): DummyQuery(field_paths={"output_video": output_video}, count=1),
        }
    )

    with VideoFetcher(video_table=table) as fetcher:
        fetched = fetcher.fetch_video(restriction, attachment_field="output_video")
        assert fetched == output_video

    assert not output_video.exists()


def test_video_fetcher_fetch_videos_cleanup_when_later_fetch_raises(tmp_path):
    first_video = tmp_path / "video1.mp4"
    first_video.write_text("data")
    key_1 = {"video_id": 1}
    key_2 = {"video_id": 2}
    table = DummyVideoTable(
        {
            repr(key_1): DummyQuery(first_video, count=1),
            repr(key_2): DummyQuery(count=1, fetch_exc=RuntimeError("fetch failed")),
        }
    )

    with pytest.raises(RuntimeError, match="fetch failed"):
        with VideoFetcher(video_table=table) as fetcher:
            fetcher.fetch_videos([key_1, key_2])

    assert not first_video.exists()
