import torch
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") == "0":
##### NOTE: This line is neccessary because SadTalker code assumes that you always want to use CUDA if is_available()
    torch.cuda.is_available = lambda : False

from src.utils.preprocess import CropAndExtract
from src.utils.init_path import init_path
import argparse
from PIL import Image
from torchvision import transforms
import cv2  # Importing OpenCV for handling video files
import tempfile

def get_num_of_audio_frames(audio_file):
    import librosa
    from src.generate_batch import parse_audio_length
    audio_data = librosa.core.load(audio_file, sr=16000)[0]
    return parse_audio_length(len(audio_data), sr=16000, fps=25)[1]

def extract_frames(video_path, output_dir, target_frame_count=None):
    """Extracts all frames from the given video file and saves them in the specified directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        if target_frame_count is not None:
            os.system(f"ffmpeg -i {video_path} -hide_banner -loglevel error -frames:v {target_frame_count} {temp_dir}/out.mp4")
            video_path = f"{temp_dir}/out.mp4"
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()  
        count = 0
        frames = []
        while success:
            frame_path = os.path.join(output_dir, f'frame{count:06d}.jpg')
            cv2.imwrite(frame_path, image)  # Save frame as JPEG file
            frames.append(frame_path)
            success, image = vidcap.read()
            count += 1
        vidcap.release()
        return frames

def test_crop_image(pic_path, preprocess_model, preprocess_option, pic_size):
    first_frame_dir = 'test_output'  # Directory where the output of the test will be saved
    os.makedirs(first_frame_dir, exist_ok=True)

    # Generate cropped image and coefficients
    first_coeff_path, crop_pic_path, crop_info = preprocess_model.generate(
        pic_path, first_frame_dir, preprocess_option, source_image_flag=True, pic_size=pic_size)

    if first_coeff_path is None:
        print("Crop and extraction failed: Could not obtain coefficients.")
    else:
        print(f"Cropped Image saved at: {crop_pic_path}")
        print(f"3DMM Coefficients Path: {first_coeff_path}")
        print(f"Cropping Information: {crop_info}")
    return crop_pic_path

def load_and_process_images(image_paths, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load images and apply transformation
    images = [transform(Image.open(path).convert('RGB')) for path in image_paths]
    image_tensor = torch.stack(images)  # Shape: (N, 3, 224, 224)
    return image_tensor.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default='./examples/happy/happy_original.mp4', help="Path to the video file")
    parser.add_argument("--output_dir", default='frames', help="Directory to save extracted frames")
    parser.add_argument("--output-betas", default=None, help="File to save extracted betas")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="How to preprocess the images")
    parser.add_argument("--size", type=int, default=224, help="The image size for face rendering")
    parser.add_argument("--device", default='cpu', help="Device to run the cropping on, e.g., 'cpu' or 'cuda'")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="Path to the model checkpoints")
    parser.add_argument("--config_dir", default='./src/config', help="Path to the configuration directory")
    parser.add_argument("--old_version", action='store_true', help="Flag to use the old version of the models")

    args = parser.parse_args()

    # Initialize paths using the provided init_path function
    sadtalker_paths = init_path(args.checkpoint_dir, args.config_dir, args.size, args.old_version, args.preprocess)

    # Create directory for frames if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    preprocess_model = CropAndExtract(sadtalker_paths, args.device)
    # preprocess_model.net_recon = preprocess_model.net_recon.to(args.device)
    # preprocess_model.propress.predictor.det_net = preprocess_model.propress.predictor.det_net.to(args.device)
    # preprocess_model.propress.predictor.det_net.backbone = preprocess_model.propress.predictor.det_net.backbone.to(args.device)
    
    # Extract frames from the video
    frames = extract_frames(args.video_path, args.output_dir)

    from src.utils.model2safetensor import net_recon

    net_recon = net_recon.to(args.device)

    # Prepare the cropped images
    cropped_images = [test_crop_image(frame, preprocess_model, args.preprocess, args.size) for frame in frames]
    cropped_tensor = load_and_process_images(cropped_images, args.device)
    output_tensor = net_recon(cropped_tensor)
    # crop tensor to only expressions
    output_tensor = output_tensor[:, 80:144]
    print("Output tensor shape:", output_tensor.shape)
    if args.output_betas:
        torch.save(output_tensor, args.output_betas)
