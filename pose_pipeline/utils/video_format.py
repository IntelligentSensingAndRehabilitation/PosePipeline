from pose_pipeline.pipeline import Video
import subprocess
import tempfile
import os
from pathlib import Path


def compress(fn, bitrate=5):
    import subprocess

    fd, temp = tempfile.mkstemp(suffix=".mp4")
    subprocess.run(["ffmpeg", "-y", "-i", fn, "-c:v", "libx264", "-b:v", f"{bitrate}M", "-fps_mode", "vfr", temp])
    os.close(fd)
    return temp


def insert_local_video(filename, video_start_time, local_path, video_project="TESTING", skip_duplicates=False):
    """Insert local video into the Pose Pipeline"""

    assert os.path.exists(local_path)

    vid_struct = {
        "video_project": video_project,
        "filename": filename,
        "start_time": video_start_time,
        "video": local_path,
    }

    print(vid_struct)
    Video().insert1(vid_struct, skip_duplicates=skip_duplicates)


def make_browser_friendly(
    filename,
    base_dir,
    backup_folder_name="Original Browser Incompatible Videos",
    crf=18,
    preset="fast"
):
    """
    Used to pre-process videos taken from non-lab standard cameras to ensure they visualize correctly in browsers and jupyter notebooks

    Steps:
    1. Moves the original file into a backup folder (e.g. 'Original Browser Incompatible Videos').
    2. Run ffmpeg on the backup file.
    3. Write the transcoded output back to the original directory with the original filename.
    """
    base_dir = Path(base_dir)
    filename = Path(filename).name  # ensure we're only using the name, not any stray path
    original_path = base_dir / filename

    if not original_path.exists():
        raise FileNotFoundError(f"Video file not found: {original_path}")

    # Create backup directory
    backup_dir = base_dir / backup_folder_name
    backup_dir.mkdir(exist_ok=True)

    # Move original file to backup folder
    backup_path = backup_dir / filename
    print(f"Moving original file:\n  {original_path}\n→ {backup_path}")
    original_path.rename(backup_path)

    # The output will be written back to the original location with the original name
    output_path = original_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(backup_path),          # read from the backup (original file)
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",           # critical for browser support
        "-movflags", "+faststart",       # put moov atom at front for streaming
        "-c:a", "aac",
        "-b:a", "128k",
        str(output_path),
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"Transcoded video written to: {output_path}")
    return str(output_path), str(backup_path)
