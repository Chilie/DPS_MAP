python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/sr4_config.yaml --save_dir=ffhq_DDIM_100_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/imagenet_sr4_config.yaml --save_dir=imagenet_DDIM_100_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/sr4_config.yaml --save_dir=ffhq_DDPM_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/imagenet_sr4_config.yaml --save_dir=imagenet_DDPM_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/color_config.yaml --save_dir=ffhq_DDIM_100_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/imagenet_color_config.yaml --save_dir=imagenet_DDIM_100_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/color_config.yaml --save_dir=ffhq_DDPM_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/imagenet_color_config.yaml --save_dir=imagenet_DDPM_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/deblur_gauss_config.yaml --save_dir=ffhq_DDIM_100_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/imagenet_deblur_gauss_config.yaml --save_dir=imagenet_DDIM_100_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/deblur_gauss_config.yaml --save_dir=ffhq_DDPM_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/imagenet_deblur_gauss_config.yaml --save_dir=imagenet_DDPM_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/deblur_uniform_config.yaml --save_dir=ffhq_DDIM_100_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config_DDIM.yaml --task_config=configs_poisson/imagenet_deblur_uniform_config.yaml --save_dir=imagenet_DDIM_100_Poisson

python3 main_Poisson_batched.py --model_config=configs_poisson/model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/deblur_uniform_config.yaml --save_dir=ffhq_DDPM_Poisson
python3 main_Poisson_batched.py --model_config=configs_poisson/imagenet_model_config.yaml --diffusion_config=configs_poisson/diffusion_config.yaml --task_config=configs_poisson/imagenet_deblur_uniform_config.yaml --save_dir=imagenet_DDPM_Poisson