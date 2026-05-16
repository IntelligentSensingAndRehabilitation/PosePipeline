import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def cleanup_video_files(paths):
    """Remove local video files fetched from DataJoint attachments."""
    for path in paths:
        file_path = Path(path)
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
        except Exception as exc:
            logger.warning(f"Failed to clean up video file {file_path}: {exc}")


class VideoFetcher:
    """Context manager for fetching and cleaning up local video files."""

    def __init__(self, video_table=None):
        if video_table is None:
            from pose_pipeline.pipeline import Video

            video_table = Video
        self.video_table = video_table
        self._tracked_videos = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_video_files(self._tracked_videos)
        return False

    def fetch_video(self, restriction):
        """Fetch one video path from a key dict or a restricted Video relation."""
        video_query = self.video_table & restriction

        if len(video_query) != 1:
            raise ValueError(
                f"Video query must match exactly 1 video, found {len(video_query)} "
                f"for restriction: {restriction}"
            )

        video_path = Path(video_query.fetch1("video"))
        self._tracked_videos.append(video_path)
        return video_path

    def fetch_videos(self, restrictions):
        """Fetch multiple videos, tracking all returned local paths for cleanup."""
        return [self.fetch_video(restriction) for restriction in restrictions]
