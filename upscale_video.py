import compat  # Fix for torchvision bug

import os
import cv2
import torch
import ffmpeg
import argparse
import subprocess
import numpy as np
import urllib.request
from tqdm import tqdm
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet


# --------------------------- #
#        Model Handling       #
# --------------------------- #
def download_model_if_missing(model_name):
    model_urls = {
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }
    model_path = os.path.join("models", model_name)
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        print(f"Model '{model_name}' not found. Downloading...")
        if model_name not in model_urls:
            raise ValueError(f"No download URL for model '{model_name}'")
        urllib.request.urlretrieve(model_urls[model_name], model_path)
        print("✅ Model downloaded.")
    return model_path


def create_upsampler(scale, model_path, device):
    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )
    return RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=384, #Increase to make the script faster, decrease if you're having problems. The number should be divisible by 64
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device
    )


# --------------------------- #
#         Stream Mode         #
# --------------------------- #
def stream_upscale(input_path, output_path, scale, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = download_model_if_missing(model_name)
    upsampler = create_upsampler(scale, model_path, device)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

    ffmpeg_proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', '-', '-an', '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ], stdin=subprocess.PIPE)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        output, _ = upsampler.enhance(frame)
        ffmpeg_proc.stdin.write(output.astype(np.uint8).tobytes())
        frame_count += 1

    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    print(f"✅ Stream upscale complete! {frame_count} frames processed.")


# --------------------------- #
#     Disk-Based (Legacy)     #
# --------------------------- #
def extract_frames(video_path, frame_dir):
    os.makedirs(frame_dir, exist_ok=True)
    (
        ffmpeg
        .input(video_path)
        .output(f'{frame_dir}/frame_%06d.png', qscale=2)
        .run()
    )


def upscale_frames(input_dir, output_dir, scale, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = download_model_if_missing(model_name)
    upsampler = create_upsampler(scale, model_path, device)

    os.makedirs(output_dir, exist_ok=True)
    frame_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".png"))

    for frame_file in tqdm(frame_files, desc="Upscaling frames"):
        input_path = os.path.join(input_dir, frame_file)
        output_path = os.path.join(output_dir, frame_file)
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"⚠️ Skipping unreadable frame: {input_path}")
            continue
        output, _ = upsampler.enhance(image)
        cv2.imwrite(output_path, output)


def combine_frames(frame_dir, output_video, fps=30):
    (
        ffmpeg
        .input(f'{frame_dir}/frame_%06d.png', framerate=fps)
        .output(output_video, vcodec='libx264', pix_fmt='yuv420p')
        .run()
    )


def cleanup_frames(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name != ".gitkeep":
            os.remove(file_path)


def upscale_with_disk(input_path, output_path, scale, model_name):
    input_frames = 'frames/input'
    output_frames = 'frames/output'

    print('Extracting frames...')
    extract_frames(input_path, input_frames)

    print('Upscaling frames...')
    upscale_frames(input_frames, output_frames, scale, model_name)

    print('Combining frames into video...')
    combine_frames(output_frames, output_path)

    print('Cleaning up temporary frames...')
    cleanup_frames(input_frames)
    cleanup_frames(output_frames)

    print('✅ Legacy (disk) pipeline complete!')


# --------------------------- #
#             CLI             #
# --------------------------- #
def main():
    parser = argparse.ArgumentParser(description='AI Video Upscaling')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', required=True, help='Path for output video')
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help='Upscaling factor')
    parser.add_argument('--model', default=None, help='Model name in models/ folder or auto-download')
    parser.add_argument('--legacy', action='store_true', help='Use disk-based frame processing')

    args = parser.parse_args()

    if not args.model:
        args.model = f"RealESRGAN_x{args.scale}plus.pth"

    if args.legacy:
        print("🧱 Using legacy disk-based pipeline...")
        upscale_with_disk(args.input, args.output, args.scale, args.model)
    else:
        print("⚡ Using fast in-memory stream mode...")
        stream_upscale(args.input, args.output, args.scale, args.model)


if __name__ == '__main__':
    main()
