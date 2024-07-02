_base_ = ['../PixArt_xl2_internal.py']
image_list_json = ['data_info.json']

data = dict(
    type='InternalDataMSSigma', root='InternData', image_list_json=image_list_json, transform='default_train',
    load_vae_feat=True, load_t5_feat=True
)
image_size = 512
validation_prompts = [
    "an anime woman staring directly into camera looking stoic, she is wearing silver chrome and gold armor",
    "an anime man in front of a basket of fruit"
]
deterministic_validation = True

# model setting
model = 'PixArtMS_XL_2'
mixed_precision = 'auto'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
load_from = "output/pretrained_models/PixArt-Sigma-XL-2-512-MS.pth"  # https://huggingface.co/PixArt-alpha/PixArt-Sigma
resume_from = None
vae_pretrained = "output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers/vae"  # sdxl vae
aspect_ratio_type = 'ASPECT_RATIO_512'
multi_scale = True  # if use multiscale dataset model training
pe_interpolation = 1.0

# training setting
num_workers = 10
train_batch_size = 3  # 3 for w.o feature extraction; 12 for feature extraction
num_epochs = 9 # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='CAMEWrapper', lr=1e-5, weight_decay=0.0, betas=(0.9, 0.999, 0.9999), eps=(1e-30, 1e-16))
lr_schedule_args = dict(num_warmup_steps=50)

eval_sampling_steps = 500_000_000
visualize = True
log_interval = 6
save_model_epochs = 1
save_model_steps = 100_000_000
work_dir = 'output/debug'

# pixart-sigma
scale_factor = 0.13025
real_prompt_ratio = 0.5
model_max_length = 300
class_dropout_prob = 0.1

qk_norm = False
skip_step = 0  # skip steps during data loading
