
model_setting = {
    'model_name': 'SSIT',
    'lambda_adv': {'default':1.0},
    'lambda_cyc': {'default':1.0, 'photo2art':1.0, 'BDD100K_weather&time':1.0},
    'lambda_style': {'default':5.0},

    'k_shot': None,
    'img_size': {'default':(256, 256), 
                 'BDD100K_weather&time':(256, 512)},
    'pad_crop': 32,

    'unroll_steps': None,
    }

netG_params = {
    'base_dim': 32,
    'enc_dim_num': 6,
    }

netD_params = {
    'base_dim': 32,
    'n_layers': 4,
    'netD_num': 1,
    }