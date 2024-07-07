from tqdm import tqdm
import torch
from torch import nn
import safetensors 
import safetensors.torch 
from time import strftime
import os, sys
from argparse import ArgumentParser
import pathlib

from src.utils.preprocess import CropAndExtract
from src.generate_batch import get_data
from src.utils.init_path import init_path
from src.utils.safetensor_helper import load_x_from_safetensor 
from src.audio2exp_models.networks import SimpleWrapperV2
from yacs.config import CfgNode as CN


def load_cpk(checkpoint_path, model=None, optimizer=None, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    if model is not None:
        model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint['epoch']

class Audio2Exp(nn.Module):
    def __init__(self, netG, cfg, device, prepare_training_loss=False):
        super(Audio2Exp, self).__init__()
        self.cfg = cfg
        self.device = device
        self.netG = netG.to(device)

    def forward(self, batch):

        mel_input = batch['indiv_mels']                         # bs T 1 80 16
        bs = mel_input.shape[0]
        T = mel_input.shape[1]

        exp_coeff_pred = []

        for i in tqdm(range(0, T, 10),'audio2exp:'): # every 10 frames
            
            current_mel_input = mel_input[:,i:i+10]

            #ref = batch['ref'][:, :, :64].repeat((1,current_mel_input.shape[1],1))           #bs T 64
            ref = batch['ref'][:, :, :64][:, i:i+10]
            ratio = batch['ratio_gt'][:, i:i+10]                               #bs T  Zblink

            audiox = current_mel_input.view(-1, 1, 80, 16)                  # bs*T 1 80 16

            curr_exp_coeff_pred  = self.netG(audiox, ref, ratio)         # bs T 64 

            exp_coeff_pred += [curr_exp_coeff_pred]

        # BS x T x 64
        results_dict = {
            'exp_coeff_pred': torch.cat(exp_coeff_pred, axis=1)  #1 200 64
            }
        return results_dict



class Audio2Exp_pretrained(nn.Module):
    def __init__(self, sadtalker_path, device):
        super(Audio2Exp_pretrained, self).__init__()
        fcfg_exp = open(sadtalker_path['audio2exp_yaml_path'])
        cfg_exp = CN.load_cfg(fcfg_exp)

        self.device = device
        self.audio2exp_model = self.initialize_model(sadtalker_path, cfg_exp, device)

    def initialize_model(self, sadtalker_path, cfg_exp, device):
        netG = SimpleWrapperV2().to(device)

        try:
            if sadtalker_path['use_safetensor']:
                checkpoints = safetensors.torch.load_file(sadtalker_path['checkpoint'])
                netG.load_state_dict(load_x_from_safetensor(checkpoints, 'audio2exp'))
            else:
                load_cpk(sadtalker_path['audio2exp_checkpoint'], model=netG, device=device)
        except:
            raise Exception("Failed in loading audio2exp_checkpoint")

        self.audio2exp_model = Audio2Exp(netG, cfg_exp, device=device, prepare_training_loss=False)
        self.audio2exp_model = self.audio2exp_model.to(device)

        return self.audio2exp_model

    def generate(self, batch):    
        results_dict_exp= self.audio2exp_model(batch)
        exp_pred = results_dict_exp['exp_coeff_pred']
        return exp_pred

if __name__ == "__main__":
    parser = ArgumentParser("training_loop")
    parser.add_argument("--driven_audios", type=pathlib.Path)
    parser.add_argument("--source_images", type=pathlib.Path)
    parser.add_argument("--betas", type=pathlib.Path)
    parser.add_argument("--checkpoint_dir", type=pathlib.Path, default='./checkpoints')
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--output", type=pathlib.Path, default='./checkpoints/fine_tuned_expnet_happy.pth')
    parser.add_argument("--blink_dir", type=pathlib.Path, default='./checkpoints')
    parser.add_argument("-e", "--epochs", type=int, default=450)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    parser.add_argument("--size", type=int, default=256,  help="the image size of the facerender")
    parser.add_argument("--still", action="store_true", help="can crop back to the original videos for the full body aniamtion")
    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images" )
    args = parser.parse_args()

    audios = args.driven_audios.absolute()
    imgs = args.source_images.absolute()
    betas = args.betas.absolute()

    audios_sorted = sorted(os.listdir(audios))
    imgs_sorted = sorted(os.listdir(imgs))
    betas_sorted = sorted(os.listdir(betas))
    epochs = args.epochs
    batch = args.batch_size

    current_root_path = os.path.split(sys.argv[0])[0]
    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size, args.old_version, args.preprocess)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    preprocess_model = CropAndExtract(sadtalker_paths, device)
    audio2exp_model = Audio2Exp_pretrained(sadtalker_paths,  device)
    n_obs = len(audios_sorted)

    save_dir = os.path.join(args.blink_dir, strftime("%Y_%m_%d_%H.%M.%S"))

    first_frame_dir = 'audio2betas_test'
    os.makedirs(first_frame_dir, exist_ok=True)

    if args.ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(args.ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(args.ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    optimizer = torch.optim.Adam(audio2exp_model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        audio2exp_model.train()
        running_loss = 0.0

        for i in range(0, n_obs, batch):
            batch_audios = audios_sorted[i:i + batch]
            batch_imgs = imgs_sorted[i:i + batch]
            batch_betas = betas_sorted[i:i + batch]
            batch_audios = [os.path.join(audios, file_name) for file_name in batch_audios]
            batch_imgs = [os.path.join(imgs, file_name) for file_name in batch_imgs]
            batch_betas = [os.path.join(betas, file_name) for file_name in batch_betas]

            optimizer.zero_grad()

            for j in range(min(batch, n_obs-i*batch)):
                first_coeff_path, _, _ =  preprocess_model.generate(batch_imgs[j], first_frame_dir, args.preprocess, source_image_flag=True, pic_size=args.size)
                data = get_data(first_coeff_path, batch_audios[j], device, ref_eyeblink_coeff_path, still=args.still)
                output = audio2exp_model.generate(data).to(device)
                betas_batch = torch.load(batch_betas[j]).unsqueeze(dim=0).to(device)

                loss = criterion(output, betas_batch)
                running_loss += loss.item()
                loss.backward()

            optimizer.step()

        epoch_loss = running_loss / (n_obs // batch)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}')

    checkpoint = {
        'epoch': epochs,
        'model': audio2exp_model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, args.output)