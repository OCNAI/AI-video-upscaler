# AI Video Upscaler

The AI Video Upscaler is a Python-based local video upscaling solution leveraging state-of-the-art AI image upscaling algorithms. 
This project splits videos into individual frames, enhances each frame using AI image enhancement techniques, and recombines the frames into a high-quality upscaled video, entirely on your local machine.

## Features:

* __Local AI-based video upscaling:__ No reliance on cloud services or paid APIs.
* __High-quality frame interpolation:__ Uses Python and FFMPEG for lossless video-to-frame and frame-to-video conversions.
* __Simple CLI:__ Easy-to-use command line interface for quick video processing.

## Installation:
### Prerequisites:
Ensure the following software is installed and available in your system PATH:
* Python 3.10 or later
* [FFMPEG](https://www.ffmpeg.org/)

### Setup

1. Clone the repository:

    git clone https://github.com/YourUsername/ai-video-upscaler.git
    cd ai-video-upscaler

2. Create a virtual environment and activate it:

    python -m venv venv
    
    source venv/bin/activate  # Linux/Mac

    .\venv\Scripts\activate  # Windows

3. Install dependencies:

    pip install -r requirements.txt

## Usage: 

1. Place the video you wish to upscale into the input_videos/ folder.
2. Run the upscale script:

python upscale_video.py --input input_videos/your_video.mp4 --output output_videos/upscaled_video.mp4

### Optional arguments:

--scale: Upscaling factor (default is 2 for 2x upscale).

--model: Choose a specific model (default is RealESRGAN_x4plus).

## Recommended hardware specs:
To efficiently use this software, we recommend the following specifications:
* __CPU:__ 
    * __Minimum (Can Work):__ Quad-core (i5 or Ryzen 5)
    * __Recommended:__ 6+ cores (i7, Ryzen 7+)
* __GPU:__
    * __Minimum (Can Work):__ NVIDIA GTX 1060 6GB (Or equivalent)
    * __Recommended:__ NVIDIA RTX 3060 / 3070 / 3080+ 8GB–12GB VRAM (Or equivalent)
* __RAM:__
    * __Minimum (Can Work):__ 16GB DDR4
    * __Recommended:__ 32GB or more DDR5
* __Storage:__
    * __Minimum (Can Work):__ HDD with enough space for your upscaled video
    * __Recommended:__ NVMe SSD with enough space for your upscaled video

## Repository structure explained:
ai-video-upscaler/

├── input_videos/            # Folder for input videos

├── output_videos/           # Folder for output upscaled videos

├── frames/                  # Temporary folder for extracted frames

│   ├── input/

│   └── output/

├── models/                  # Folder for AI models

├── venv/                    # Python virtual environment

├── upscale_video.py         # Main Python script

├── requirements.txt         # Project dependencies

├── .gitignore               

└── README.md

## License:
None yet