import os
import glob
import shutil
import subprocess
import time
import tempfile
import bz2
import zipfile
from multiprocessing import Pool, cpu_count

# Added for automatic downloads
try:
    import requests
except ImportError:
    print("Error: The 'requests' library is not installed. Please install it using: pip install requests")
    exit()

import cv2
import dlib
import numpy as np
from textgrid import TextGrid, IntervalTier
from tqdm import tqdm

# --- Helper functions for multiprocessing (must be at the top level) ---

_dlib_detector = None
_dlib_predictor = None

def _init_dlib_worker(predictor_path):
    """Initializes dlib models for each worker process."""
    global _dlib_detector, _dlib_predictor
    _dlib_detector = dlib.get_frontal_face_detector()
    _dlib_predictor = dlib.shape_predictor(predictor_path)

def _process_video_worker(args):
    """The actual workhorse function for video processing."""
    video_path, dataset_root, output_root, video_height, video_width = args
    cap = None
    try:
        relative_path = os.path.relpath(video_path, dataset_root)
        base_relative_path, _ = os.path.splitext(relative_path)
        output_path = os.path.join(output_root, base_relative_path + '.mp4')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return (video_path, "Error: OpenCV could not open video.", 0)
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 25.0

            last_known_bbox = None
            frames_saved = 0
            frame_index = 0

            while True:
                ret, frame_bgr = cap.read()
                if not ret: break

                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                faces = _dlib_detector(gray, 1)
                
                current_bbox = None
                if faces:
                    landmarks = _dlib_predictor(frame_bgr, faces[0])
                    mouth_points = np.array([[p.x, p.y] for p in landmarks.parts()[48:68]])
                    x_min, y_min = np.min(mouth_points, axis=0)
                    x_max, y_max = np.max(mouth_points, axis=0)
                    padding = 10
                    new_bbox = (
                        max(0, int(x_min - padding)), max(0, int(y_min - padding)),
                        min(frame_bgr.shape[1], int(x_max + padding)), min(frame_bgr.shape[0], int(y_max + padding))
                    )
                    temp_crop = frame_bgr[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
                    if temp_crop.size > 0:
                        current_bbox = new_bbox
                        last_known_bbox = new_bbox
                
                if current_bbox is None:
                    if last_known_bbox is not None: current_bbox = last_known_bbox
                    elif frame_index == 0: return (video_path, "Error: No valid face on first frame.", 0)

                if current_bbox:
                    mouth_crop = frame_bgr[current_bbox[1]:current_bbox[3], current_bbox[0]:current_bbox[2]]
                    if mouth_crop.size > 0:
                        resized_crop = cv2.resize(mouth_crop, (video_width, video_height))
                        image_path = os.path.join(temp_dir, f"frame_{frames_saved:05d}.png")
                        cv2.imwrite(image_path, resized_crop)
                        frames_saved += 1
                
                frame_index += 1

            if frames_saved > 0:
                ffmpeg_command = [
                    'ffmpeg', '-y', '-framerate', str(fps),
                    '-i', os.path.join(temp_dir, 'frame_%05d.png'),
                    '-c:v', 'libx264', '-r', str(fps), '-pix_fmt', 'yuv420p',
                    output_path
                ]
                process = subprocess.run(ffmpeg_command, capture_output=True, text=True, encoding='utf-8')
                if process.returncode != 0:
                    return (video_path, f"FFmpeg Error: {process.stderr}", 0)
            
            return (video_path, "Success", frames_saved)

    except Exception as e:
        return (video_path, f"Python Error: {e}", 0)
    finally:
        if cap: cap.release()


class Aligner:
    """A comprehensive pipeline for processing lip-reading datasets."""

    def __init__(self, input_dir, output_dir):
        """
        Initializes the Aligner pipeline. Dependencies are handled automatically.

        Args:
            input_dir (str): Path to the root directory containing source videos and text files.
            output_dir (str): Path to the root directory where all processed files will be saved.
        """
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        
        # --- Define all output and dependency directories ---
        self.dependencies_dir = os.path.join(self.output_dir, "_dependencies")
        self.dlib_dir = os.path.join(self.dependencies_dir, "dlib")
        self.mfa_dir = os.path.join(self.dependencies_dir, "mfa")
        
        self.mouth_videos_dir = os.path.join(self.output_dir, "MouthVideos")
        self.audio_lab_dir = os.path.join(self.output_dir, "audio_lab")
        self.textgrid_dir = os.path.join(self.output_dir, "TextGrid")
        self.word_align_dir = os.path.join(self.output_dir, "Word_align")
        self.phoneme_align_dir = os.path.join(self.output_dir, "Phoneme_align")
        self.phone_word_align_dir = os.path.join(self.output_dir, "Phone_Word_align")

        # --- Define expected paths for dependency files ---
        self.dlib_predictor_path = os.path.join(self.dlib_dir, "shape_predictor_68_face_landmarks.dat")
        self.mfa_model_path = os.path.join(self.mfa_dir, "japanese_mfa.zip")
        self.mfa_dict_path = os.path.join(self.mfa_dir, "japanese_mfa.dict")
        
        self.SAMPLING_RATE = 16000
        self.SILENCE_TOKEN = "sil"

    def _download_file(self, url, dest_path):
        """Downloads a file with a progress bar."""
        print(f"Downloading {os.path.basename(url)}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                with open(dest_path, 'wb') as f, tqdm(
                    total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            return False

    def _setup_dependencies(self):
        """Checks for, downloads, and unpacks all required external files."""
        print("--- Checking and setting up dependencies ---")
        os.makedirs(self.dlib_dir, exist_ok=True)
        os.makedirs(self.mfa_dir, exist_ok=True)

        # 1. Check for FFMPEG
        if not shutil.which("ffmpeg"):
            raise FileNotFoundError("FATAL ERROR: FFmpeg is not installed or not in your system's PATH.")
        
        # 2. Setup Dlib predictor
        if not os.path.exists(self.dlib_predictor_path):
            print("Dlib shape predictor not found.")
            dlib_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
            bz2_path = os.path.join(self.dlib_dir, os.path.basename(dlib_url))
            if self._download_file(dlib_url, bz2_path):
                print("Decompressing predictor...")
                with bz2.open(bz2_path, 'rb') as f_in, open(self.dlib_predictor_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(bz2_path)
                print("Dlib predictor is ready.")

        # 3. Setup MFA model and dictionary
        if not os.path.exists(self.mfa_dict_path):
            print("MFA model/dictionary not found.")
            mfa_url = "https://github.com/Montreal-Forced-Aligner/mfa-models/releases/download/acoustic-v2.0.0/japanese_mfa.zip"
            if self._download_file(mfa_url, self.mfa_model_path):
                print("Unzipping MFA model...")
                with zipfile.ZipFile(self.mfa_model_path, 'r') as zip_ref:
                    zip_ref.extractall(self.mfa_dir)
                # The .zip file itself is the model path MFA uses. The .dict is now extracted.
                if not os.path.exists(self.mfa_dict_path):
                    # Attempt to find the dict file if it has a different name
                    found_dicts = glob.glob(os.path.join(self.mfa_dir, '*.dict'))
                    if found_dicts:
                        # Rename the first found dict to the expected name
                        os.rename(found_dicts[0], self.mfa_dict_path)
                        print("MFA dictionary is ready.")
                    else:
                        raise FileNotFoundError("Could not find .dict file in the downloaded MFA model zip.")
                else:
                    print("MFA model and dictionary are ready.")

        print("All dependencies are ready.")

    # ... (the rest of the class methods: _create_dirs, _prepare_audio_for_mfa, etc. remain the same) ...
    def _create_dirs(self):
        print("--- Creating output directory structure ---")
        for path in [self.mouth_videos_dir, self.audio_lab_dir, self.textgrid_dir,
                     self.word_align_dir, self.phoneme_align_dir, self.phone_word_align_dir]:
            os.makedirs(path, exist_ok=True)

    def _prepare_audio_for_mfa(self):
        print("\n[PIPELINE STEP 1/5] Preparing WAV and LAB files for MFA...")
        video_files = glob.glob(os.path.join(self.input_dir, '**', '*.mp4'), recursive=True)
        if not video_files: print("Warning: No .mp4 files found."); return

        for video_path in tqdm(video_files, desc="Preparing audio/lab"):
            relative_path = os.path.relpath(video_path, self.input_dir)
            base_name, _ = os.path.splitext(relative_path)
            lab_path = os.path.join(self.audio_lab_dir, base_name + '.lab')
            wav_path = os.path.join(self.audio_lab_dir, base_name + '.wav')
            os.makedirs(os.path.dirname(lab_path), exist_ok=True)

            if os.path.exists(lab_path) and os.path.exists(wav_path): continue
            
            txt_path = os.path.splitext(video_path)[0] + '.txt'
            if not os.path.exists(txt_path):
                tqdm.write(f"Warning: Skipping '{video_path}' (no .txt file).")
                continue

            try:
                ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', str(self.SAMPLING_RATE), '-ac', '1', '-y', wav_path]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                shutil.copyfile(txt_path, lab_path)
            except subprocess.CalledProcessError as e:
                tqdm.write(f"FFMPEG ERROR on {video_path}. Skipping. Stderr: {e.stderr.decode('utf-8', errors='ignore')}")

    def _run_mfa_align(self):
        print("\n[PIPELINE STEP 2/5] Running Montreal Forced Aligner...")
        if not any(os.scandir(self.audio_lab_dir)): print("Warning: audio_lab directory is empty. Skipping MFA."); return

        mfa_cmd = ['mfa', 'align', self.audio_lab_dir, self.mfa_dict_path, self.mfa_model_path, self.textgrid_dir, '--clean', '--jobs', '4']
        print("Executing: " + ' '.join(mfa_cmd))
        try:
            subprocess.run(mfa_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            print("MFA alignment completed.")
        except FileNotFoundError: print("FATAL ERROR: 'mfa' command not found. Is Conda env active?")
        except subprocess.CalledProcessError as e: print(f"--- ERROR: MFA failed! ---\nStderr: {e.stderr}")

    def _extract_mouth_videos(self):
        print("\n[PIPELINE STEP 3/5] Extracting mouth videos...")
        video_files = glob.glob(os.path.join(self.input_dir, '**', '*.mp4'), recursive=True)
        tasks = [(v, self.input_dir, self.mouth_videos_dir, 50, 100) for v in video_files if not os.path.exists(os.path.join(self.mouth_videos_dir, os.path.splitext(os.path.relpath(v, self.input_dir))[0] + '.mp4'))]
        if not tasks: print("All mouth videos already processed."); return

        print(f"Found {len(tasks)} videos to process.")
        num_processes = max(1, cpu_count() - 1)
        with Pool(processes=num_processes, initializer=_init_dlib_worker, initargs=(self.dlib_predictor_path,)) as pool:
            results = list(tqdm(pool.imap_unordered(_process_video_worker, tasks), total=len(tasks), desc="Cropping videos"))
        
        errors = [f"- {os.path.basename(p)}: {s}" for p, s, _ in results if s != "Success"]
        if errors: print("\nErrors during video processing:\n" + "\n".join(errors[:10]))

    def _convert_textgrids(self):
        print("\n[PIPELINE STEP 4/5] Creating phoneme-level alignments...")
        self._format_alignments(self.phoneme_align_dir, 'phones')
        print("\n[PIPELINE STEP 5/5] Creating word-level alignments...")
        self._format_alignments(self.word_align_dir, 'words')
        print("\n[PIPELINE BONUS] Creating phonetic-word-level alignments...")
        self._format_phone_word_alignments()

    def _format_alignments(self, output_dir, tier_name):
        textgrids = glob.glob(os.path.join(self.textgrid_dir, '**', '*.TextGrid'), recursive=True)
        if not textgrids: return
        for tg_path in tqdm(textgrids, desc=f"Formatting {tier_name}"):
            rel_path = os.path.relpath(tg_path, self.textgrid_dir)
            out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.align')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                tier = TextGrid.fromFile(tg_path).getFirst(tier_name)
                if not tier: continue
                content = [f"{int(i.minTime*self.SAMPLING_RATE)} {int(i.maxTime*self.SAMPLING_RATE)} {i.mark.strip() or self.SILENCE_TOKEN}" for i in tier if int(i.minTime*self.SAMPLING_RATE) < int(i.maxTime*self.SAMPLING_RATE)]
                with open(out_path, 'w', encoding='utf-8') as f: f.write("\n".join(content))
            except Exception as e: tqdm.write(f"Error on {os.path.basename(tg_path)} for {tier_name}: {e}")

    def _format_phone_word_alignments(self):
        textgrids = glob.glob(os.path.join(self.textgrid_dir, '**', '*.TextGrid'), recursive=True)
        if not textgrids: return
        for tg_path in tqdm(textgrids, desc="Formatting phonetic-word"):
            rel_path = os.path.relpath(tg_path, self.textgrid_dir)
            out_path = os.path.join(self.phone_word_align_dir, os.path.splitext(rel_path)[0] + '.align')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                tg = TextGrid.fromFile(tg_path)
                word_tier, phone_tier = tg.getFirst('words'), tg.getFirst('phones')
                if not word_tier or not phone_tier: continue
                content = []
                for w_interval in word_tier:
                    start, end = int(w_interval.minTime*self.SAMPLING_RATE), int(w_interval.maxTime*self.SAMPLING_RATE)
                    if start >= end: continue
                    if not w_interval.mark.strip():
                        content.append(f"{start} {end} {self.SILENCE_TOKEN}")
                    else:
                        phonemes = [p.mark.strip() for p in phone_tier if p.minTime >= w_interval.minTime and p.maxTime <= w_interval.maxTime and p.mark.strip()]
                        if phonemes: content.append(f"{start} {end} {' '.join(phonemes)}")
                with open(out_path, 'w', encoding='utf-8') as f: f.write("\n".join(content))
            except Exception as e: tqdm.write(f"Error on {os.path.basename(tg_path)} for phonetic-word: {e}")

    def run(self):
        """Executes the entire data processing pipeline in the correct order."""
        start_time = time.time()
        print("--- Starting Full Data Processing Pipeline ---")
        
        self._setup_dependencies()
        self._create_dirs()
        self._prepare_audio_for_mfa()
        self._run_mfa_align()
        self._extract_mouth_videos()
        self._convert_textgrids()

        print(f"\n--- Pipeline Finished! ---")
        print(f"Total execution time: {(time.time() - start_time) / 60:.2f} minutes.")
        print(f"All outputs have been saved to: {self.output_dir}")
