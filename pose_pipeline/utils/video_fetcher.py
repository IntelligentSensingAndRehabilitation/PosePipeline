from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import Any


def cleanup_video_files(paths: Iterable[str | PathLike[str] | Path]) -> None:
    """Remove local video files fetched from DataJoint attachments."""
    for path in paths:
        file_path = Path(path)
        file_path.unlink()


class VideoFetcher:
    """Context manager for fetching and cleaning up local video files."""

    def __init__(self, video_table: Any = None) -> None:
        if video_table is None:
            from pose_pipeline.pipeline import Video

            video_table = Video
        self.video_table = video_table
        self._tracked_videos: list[Path] = []

    def __enter__(self) -> "VideoFetcher":
        return self

    def __exit__(self, _exc_type, _exc_value, _exc_traceback) -> bool:
        cleanup_video_files(self._tracked_videos)
        return False

    def fetch_video(self, restriction: Any, attachment_field: str = "video") -> Path:
        """Fetch one attachment path from a key dict or a restricted Video relation."""
        video_query = self.video_table & restriction

        if len(video_query) != 1:
            raise ValueError(
                f"Video query must match exactly 1 video, found {len(video_query)} "
                f"for restriction: {restriction}"
            )

        video_path = Path(video_query.fetch1(attachment_field))
        self._tracked_videos.append(video_path)
        return video_path

    def fetch_videos(self, restrictions: Iterable[Any], attachment_field: str = "video") -> list[Path]:
        """Fetch multiple videos, tracking all returned local paths for cleanup."""
        return [self.fetch_video(restriction, attachment_field=attachment_field) for restriction in restrictions]
