import subprocess
import tempfile
import os
from pathlib import Path
import cv2


def _run_ffmpeg(input_path, output_path, extra_args):
    cmd = ["ffmpeg", "-y", "-i", str(input_path)] + extra_args + [str(output_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def compress(fn, bitrate=5):
    fd, temp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    _run_ffmpeg(fn, temp, ["-c:v", "libx264", "-b:v", f"{bitrate}M", "-fps_mode", "vfr"])
    return temp


def insert_local_video(filename, video_start_time, local_path, video_project="TESTING", skip_duplicates=False):
    """Insert local video into the Pose Pipeline"""
    
    from pose_pipeline.pipeline import Video
    
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
    2. Run ffmpeg on the backup file, writing to a temp file first.
    3. Replace the original path with the transcoded output on success,
       or restore the original from backup if ffmpeg fails.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise NotADirectoryError(f"base_dir does not exist or is not a directory: {base_dir}")

    filename = Path(filename).name  # ensure we're only using the name, not any stray path
    original_path = base_dir / filename

    if not original_path.exists():
        raise FileNotFoundError(f"Video file not found: {original_path}")

    # Create backup directory
    backup_dir = base_dir / backup_folder_name
    backup_dir.mkdir(exist_ok=True)

    # Move original to backup folder with '_original' appended — only one original should ever exist
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    backup_path = backup_dir / f"{stem}_original{suffix}"
    if backup_path.exists():
        raise FileExistsError(
            f"Original backup already exists at {backup_path}. "
            f"The video may have already been processed."
        )
    print(f"Moving original file:\n  {original_path}\n→ {backup_path}")
    original_path.rename(backup_path)

    # Always output as .mp4 for guaranteed browser/Jupyter compatibility
    output_path = original_path.with_suffix(".mp4")

    # Write to a temp file first; only replace the output path on success
    fd, temp_output = tempfile.mkstemp(suffix=".mp4", dir=str(base_dir))
    os.close(fd)
    temp_output_path = Path(temp_output)

    extra_args = [
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",           # critical for browser support
        "-movflags", "+faststart",       # put moov atom at front for streaming
        "-c:a", "aac",
        "-b:a", "128k",
        "-f", "mp4",
    ]

    try:
        _run_ffmpeg(backup_path, temp_output_path, extra_args)
        temp_output_path.replace(output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        if backup_path.exists() and not original_path.exists():
            backup_path.rename(original_path)
        raise

    print(f"Transcoded video written to: {output_path}")
    return str(output_path), str(backup_path)


def verify_frame_count(cap, reported_frames):
    if reported_frames <= 0:
        return 0

    # Fast path: metadata is correct if the last frame is readable
    cap.set(cv2.CAP_PROP_POS_FRAMES, reported_frames - 1)
    is_readable, _ = cap.read()
    if is_readable:
        return reported_frames

    # Verify frame 0 is readable before binary searching
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    is_readable, _ = cap.read()
    if not is_readable:
        return 0

    # Slow path: binary search to find the highest readable frame index.
    # last_good -> highest frame index confirmed readable
    # upper_bound ->  highest index that could still be readable; shrinks as frames fail
    last_good, upper_bound = 0, reported_frames - 1

    while last_good < upper_bound:
        test_frame = (last_good + upper_bound + 1) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, test_frame)
        is_readable, _ = cap.read()
        if is_readable:
            last_good = test_frame
        else:
            upper_bound = test_frame - 1

    return last_good + 1
