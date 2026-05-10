from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clip, clear_color,clear, mask_generator, center_crop
from util.logger import get_logger
from util.data_preprocessing import CenterCropLongEdge

import tqdm

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import random
import os
import time

import torchvision.utils as tvu
def inverse_data_transform(X):
    X = (X + 1.0) / 2.0
    return torch.clamp(X, 0.0, 1.0)

# for debug
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
seed_torch()

def put_poisson(data, rate=1.0):
    import numpy as np
    data = (data + 1.0) / 2.0
    data = data.clamp(0, 1)
    device = data.device
    data = data.detach().cpu()
    data = torch.from_numpy(np.random.poisson(data * 255.0 * rate) / 255.0 / rate)
    data = data * 2.0 - 1.0
    data = data.clamp(-1, 1)
    return data.float().to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='configs/model_config.yaml', type=str)
    parser.add_argument('--diffusion_config', default='configs/diffusion_config.yaml', type=str)
    parser.add_argument('--task_config', default='configs/sr4_config.yaml', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
 
    seed_torch(args.seed)
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    
    # Load model
    model = create_model(**model_config)
   
    if model_config['use_fp16']:
        model.convert_to_fp16()
    model = model.to(device)
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'], 'seed_{}'.format(args.seed))
    os.makedirs(out_path, exist_ok=True)
    # for img_dir in ['input', 'recon', 'progress', 'truth']:
    #     os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
    for sub_f in ['input','restored']:
        if not os.path.exists(os.path.join(out_path, sub_f)):
            os.makedirs(os.path.join(out_path, sub_f),exist_ok=True)
    # Prepare dataloader
    data_config = task_config['data']
    # if data_config['name'] == 'imagenet' or data_config['name'] == 'cat':
    #     transform = transforms.Compose([
    #         CenterCropLongEdge(),
    #         transforms.Resize(256),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    #     ])
    # else:
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    dataset = get_dataset(**data_config, transforms=transform)
    # loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)
    if data_config['name'] == 'imagenet':
        loader = get_dataloader(dataset, batch_size=1, num_workers=32, train=False)
    else:
        loader = get_dataloader(dataset, batch_size=4, num_workers=32, train=False)


    

    # Do Inference
    start_time  = time.time()
    Num_count = 0
    psnr_results = []
    ratio = 1
    pbar = tqdm.tqdm(loader)
    for ref_img, filenames in pbar:
        # logger.info(f"Inference for image {i}")
        # fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        Num_count += 1

        y_x = operator.forward(ref_img)
        y_n = noiser(y_x)

        # y_n = y_x + noiser.sigma * torch.randn_like(y_x)
        # y_n = put_poisson(y_x)
        
        # General-Purpose Posterior Sampling via DMPS
        DMPS_start_time = time.time()
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        # sample = sample_fn(x_start=x_start, measurement=y_n, H_funcs=H_funcs, noise_std = noiser.sigma, record=True, save_root=out_path)
        sample = sample_fn(x_start=x_start, measurement=y_n, H_funcs=operator, noise_std = noiser.sigma, record=False, save_root=out_path)
        DMPS_end_time = time.time()
        print('DMPS running time: {}'.format(DMPS_end_time - DMPS_start_time))
        psnr = peak_signal_noise_ratio(ref_img.cpu().numpy(),sample.cpu().numpy())
        psnr_results.append([psnr])
        print('PSNR: {}'.format(psnr))

        # if measure_config['operator']['name']  == 'color':
        #     y_n = y_n.reshape(1,1,model.image_size,model.image_size)
        #     plt.imsave(os.path.join(out_path, 'input', fname), clear(y_n),cmap='gray')
        # else:
        #     input_size = int(model.image_size/ratio)  
        #     y_n = y_n.reshape(1,model.in_channels,input_size,input_size)
        #     plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        # plt.imsave(os.path.join(out_path, 'truth', fname), clear_color(ref_img))
        # plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))

        # for sub_f in ['deg','restored']:
        #     if not os.path.exists(os.path.join(out_path, sub_f)):
        #             os.makedirs(os.path.join(out_path, sub_f),exist_ok=True)
        if measure_config['operator']['name']  == 'color':
            y_n = y_n.reshape(-1,1,model.image_size,model.image_size)
            # plt.imsave(os.path.join(out_path, 'input', fname), clear(y_n),cmap='gray')
        else:
            input_size = int(model.image_size/ratio)  
            y_n = y_n.reshape(-1,model.in_channels,input_size,input_size)
        for j in range(len(y_n)):
            tvu.save_image(
                inverse_data_transform(y_n[j]), os.path.join(out_path, 'input', filenames[j]+".png")
            )
        for j in range(len(sample)):
            tvu.save_image(
                inverse_data_transform(sample[j]), os.path.join(out_path, 'restored', filenames[j]+".png")
            )
    end_time = time.time()
    running_time = end_time - start_time
    save_results = np.zeros(3)
    save_results[0] = measure_config['noise']['sigma']
    save_results[1] = running_time
    save_results[2] = Num_count
    np.savetxt(os.path.join(out_path, 'saved_results.csv'),save_results)


    np.savetxt(os.path.join(out_path, 'psnr_results.csv'),np.array(psnr_results))
    print('Total # imges:{}, total  running Time: {}'.format(Num_count,end_time - start_time))
if __name__ == '__main__':
    main()
