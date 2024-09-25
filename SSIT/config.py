
ViT_name = "dino_vitb16"

model_setting = {
    'model_name': 'SSIT',
    'lambda_adv': {'default':1.0},
    'lambda_cyc': {'default':3.0},
    'lambda_style': {'default':0.1},
    'lambda_tv':1e-3,
    'g_lr': {'default':1e-4, },
    'd_lr': {'default':4e-4, },

    'k_shot': None,
    'img_size': {'default':(256, 256), 
                 'BDD10K_seg2paint':(256, 512), 
                 'BDD100K_weather&time':(256, 512)},
    'pad_crop': 32,

    'unroll_steps': None,

    'ViT_name': ViT_name,

    'gray_th': 0.01
    }

netG_params = {
    'enc_dim': [32, 64, 128, 256],
    'bottom_dim': [256, 256],
    'dec_dim': [256, 128, 64, 32],
    }

netD_params = {
    'hidden_dim': [384, 384, 384],
    'patch_size': 16,
    'head_num': 8
    }