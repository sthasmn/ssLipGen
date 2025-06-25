# ssLipGen: Automatic Lip-Reading Dataset Generation Pipeline

`ssLipGen` is a powerful, automated Python tool designed to process video and text data into a structured, multi-format dataset suitable for training lip-reading and audio-visual speech recognition models.

Given a directory of videos and their corresponding transcripts, `ssLipGen` handles the entire preprocessing pipeline, from mouth region extraction to forced alignment, generating clean, organized outputs ready for your machine learning models.

## Features

-   **Automated Dlib Model Management**: Automatically downloads the required Dlib face predictor model on the first run.
-   **Robust Video Processing**: Extracts mouth regions from videos using Dlib, with intelligent fallback for detection failures.
-   **Forced Alignment**: Utilizes the Montreal Forced Aligner (MFA) to generate accurate time-aligned transcriptions at word and phoneme levels.
-   **Multi-Format Output**: Creates multiple types of alignment files (`.align`) for maximum flexibility with different model architectures.
-   **Parallel Processing**: Leverages multiple CPU cores for efficient video cropping, significantly speeding up dataset creation.
-   **Structured Output**: Organizes all generated files into a clean, intuitive directory structure.
-   **Resumable**: Automatically skips files that have already been processed, allowing you to resume an interrupted run.

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/sthasmn/ssLipGen.git
cd ssLipGen
```
### 2. Install system dependencies:

-  **FFmpeg:** This must be installed and accessible in your system's PATH. You can install it via your system's package manager (e.g., sudo apt-get install ffmpeg on Ubuntu, brew install ffmpeg on macOS).
-  **Conda:** The Montreal Forced Aligner runs best in a Conda environment. Please install Miniconda or Anaconda.
### 3. Set up the Conda environment and install MFA:

```Bash
conda create -n mfa python=3.10 -y
conda activate mfa
conda config --add channels conda-forge
conda install -c conda-forge montreal-forced-aligner
conda install -c conda-forge spacy sudachipy sudachidict-core
```
### 4. Download MFA Pre-trained Models:

You need to download a pre-trained acoustic model and a pronunciation dictionary for your target language using MFA's command line.

For Japanese, run the following commands:
Those line will download model and dict to "Documents\MFA\" folder (Not sure about it. Confirm yourself).
```Bash
mfa model download acoustic japanese_mfa
mfa model download dictionary japanese_mfa
```
After diwnload is finished, copy them to project's MFA folder.
Note: You can change japanese_mfa to other languages supported by MFA (e.g., english_us_arpa). However, this pipeline has only been confirmed for Japanese.

### 5. Install Python packages:

Install ssLipGen and its Python dependencies directly from the repository. Make sure your mfa conda environment is active.

```Bash
pip install .
```
This will also install requests, opencv-python, dlib, numpy, textgrid, and tqdm.


## Usage
Create a Python script (e.g., run.py) and use the sslip.Aligner class. You only need to provide the input and output directories.

```Python
# run.py
import os
from sslip.aligner import Aligner

# 1. Set your input data directory.
#    It should contain your raw video/text files (e.g., in s1, s2... subfolders)
input_data_dir = r"path/to/your/raw_videos"

# 2. Set your desired output directory.
#    All processed files will be stored here.
output_data_dir = r"path/to/your/processed_dataset"

# Initialize and run the pipeline
if __name__ == '__main__':
    pipeline = Aligner(
        input_dir=input_data_dir,
        output_dir=output_data_dir,
    )
    pipeline.run()
```
Then, from your terminal (with the mfa conda environment activated), simply run your script:

```Bash
python run.py
```
The first time you run it, the Aligner will automatically download the necessary Dlib model if it's not found in the project folder.

## Input Structure
The tool will create the following structure in your specified output directory:
```
case 1
<input_data_dir>/
├── s1/
│   └── (.mp4, .txt)
├── s2/
│   └── (.mp4, .txt)
├── s3/
│   └── (.mp4, .txt)
├── s4/
│   └── (.mp4, .txt)
├── s5/
│   └── (.mp4, .txt)
└── s6/
    └── (.mp4, .txt)

case 2
<input_data_dir>/
│── (.mp4, .txt)

```

## Output Structure
The tool will create the following structure in your specified output directory:
```
<output_data_dir>/
├── MouthVideos/
│   └── (s1, s2, etc.)
├── audio_lab/
│   └── (s1, s2, etc.)
├── TextGrid/
│   └── (s1, s2, etc.)
├── Word_align/
│   └── (s1, s2, etc.)
├── Phoneme_align/
│   └── (s1, s2, etc.)
└── Phone_Word_align/
    └── (s1, s2, etc.)
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
