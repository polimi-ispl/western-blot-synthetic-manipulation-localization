import os


def directory_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in dirs and 'trainset' not in path:
            list_dirs.append(os.path.join(path, atom))

    return list_dirs


def path_find(atom, root='.'):

    list_dirs = []
    for path, dirs, files in os.walk(root):

        if atom in path:
            list_dirs.append(os.path.join(path))

    return list_dirs

# real root folders
w_blots_real_dict = {'0': '/nas/home/smandelli/w_blots/w_blots_RGB_original_4paper_patches256_N50_sg2_bestagini',
                    '1': '/nas/home/smandelli/w_blots/w_blots_RGB_original_4paper_patches256_N50_sg2_cozzolino_anteprima',
                    '2': '/nas/home/smandelli/w_blots/w_blots_RGB_original_4paper_patches256_N50_sg2_google_anteprima'}

w_blots_synthetic_dict = {'0': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_cg_bestagini_trainmasks',
                          '1': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_cg_cozzolino_trainmasks',
                          '2': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_cg_google_trainmasks',
                          '3': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_guideddiff_bestagini',
                          '4': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_guideddiff_cozzolino',
                          '5': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_guideddiff_google',
                          '6': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_pix2pix_bestagini_trainmasks',
                          '7': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_pix2pix_cozzolino_trainmasks',
                          '8': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_pix2pix_google_trainmasks',
                          '9': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_bestagini_trunc-0.7-snapshot-005200',
                          '10': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_bestagini_trunc-1-snapshot-005200',
                          '11': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_cozzolino_trunc-0.7-snapshot-004000',
                          '12': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_cozzolino_trunc-1-snapshot-004000',
                          '13': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_google_trunc-0.7-snapshot-001200',
                          '14': '/nas/home/smandelli/w_blots/synthetic/w_blots_patches256_N50_sg2_google_trunc-0.7-snapshot-001200'}

w_blots_tampered_dict = {'cleanup': '/nas/home/amanjunath/patch-detection/data/tampered_blots/cleanup',
                    'dall_e': '/nas/home/amanjunath/patch-detection/data/tampered_blots/dall_e',
                    'gimp': '/nas/home/amanjunath/patch-detection/data/tampered_blots/gimp'}

w_blots_tampered_dict_removal = {'combined': '/nas/home/amanjunath/patch-detection/data/tampered_blots/removal'}

web_images_real_dict = {'covid': '/nas/public/exchange/semafor/eval1/web_images/original/COVIDimages_splitted/pictures',
                     'milano': '/nas/public/exchange/semafor/eval1/web_images/original/milano/pictures',
                     'military': '/nas/public/exchange/semafor/eval1/web_images/original/military_webimages/pictures',
                     'siena': '/nas/public/exchange/semafor/eval1/web_images/original/siena/pictures',
                     'climate': '/nas/public/exchange/semafor/eval1/web_images/original/climate_change/pictures',
                     'napoli': '/nas/public/exchange/semafor/eval1/web_images/original/napoli/pictures'}

m3dsynth_dict = {'cyclegan': '/nas/home/smandelli/M3Dsynth_test_set/cycle',
                         'diffusion': '/nas/home/smandelli/M3Dsynth_test_set/diffusion',
                         'pix2pix': '/nas/home/smandelli/M3Dsynth_test_set/pix2pix',
                          'real': '/nas/home/smandelli/M3Dsynth_test_set/real'}

wblots_real_dict = {'resize256': '/nas/home/smandelli/Pycharm_projects/stylegan2-ada-pytorch/dataset_wblots_orig'}

# PIX2PIX
pix2pix_real_root = '/nas/home/smandelli/Pycharm_projects/pix2pix-tf/pix2pix_real_images/'
pix2pix_real_keys = next(os.walk(pix2pix_real_root))[1]
pix2pix_real_dirs = [os.path.join(pix2pix_real_root, path) for path in pix2pix_real_keys]
pix2pix_real_dict = {pix2pix_real_keys[i]: pix2pix_real_dirs[i] for i in range(len(pix2pix_real_keys))}

efros_test_root = '/nas/home/smandelli/Pycharm_projects/wifs2020/dataset/'
atom = '0_real'
efros_real_dirs = directory_find(atom, efros_test_root)
# build dictionary
efros_real_keys = ['-'.join(x.split(efros_test_root)[-1].split(atom)[0].split('/')[:-1]) for x in efros_real_dirs]
efros_real_dict = {efros_real_keys[i]: efros_real_dirs[i] for i in range(len(efros_real_keys))}

# EFROS reduced to image-to-image translation models
i2i_models = ['crn', 'cyclegan', 'gaugan', 'imle', 'stargan']
i2i_cyclegan_models = ['cyclegan']

efros_real_i2i_keys = []
efros_real_i2i_dirs = []
for model in i2i_models:
    for k_idx, k in enumerate(efros_real_keys):
        if model in k:
            efros_real_i2i_keys.append(k)
            efros_real_i2i_dirs.append(efros_real_dirs[k_idx])
efros_real_i2i_dict = {efros_real_i2i_keys[i]: efros_real_i2i_dirs[i] for i in range(len(efros_real_i2i_keys))}
efros_real_i2i_cyclegan_dict = {efros_real_i2i_keys[i]: efros_real_i2i_dirs[i] for i in range(len(efros_real_i2i_keys)) if 'cyclegan' in efros_real_i2i_keys[i]}

# stargan2 real:
stargan2_real_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/data/afhq/'
stargan2_real_init_keys = next(os.walk(stargan2_real_root))[1]
stargan2_real_dirs = []
stargan2_real_keys = []
for key in stargan2_real_init_keys:
    stargan2_real_dirs.extend(path_find(key, stargan2_real_root)[1:])
    stargan2_real_keys.extend(['-'.join(x.split('data/')[-1].split('/')) for x in path_find(key, stargan2_real_root)[1:]])
stargan2_real_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/data/celeba_hq/'
for key in stargan2_real_init_keys:
    stargan2_real_dirs.extend(path_find(key, stargan2_real_root)[1:])
    stargan2_real_keys.extend(['-'.join(x.split('data/')[-1].split('/')) for x in path_find(key, stargan2_real_root)[1:]])

stargan2_real_dict = {stargan2_real_keys[i]: stargan2_real_dirs[i] for i in range(len(stargan2_real_keys))}

ffhq_real_dict = {'ffhq': '/nas/home/nbonettini/projects/jstsp-benford-gan/additional_gan_images/ffhq'}

eval1_pristine_dict = {'eval1_pristine': '/nas/public/dataset/semafor/eval_1/selected_pristine_images'}

# primo training: P = 128, N = 1 dà TPR@0.1 = 54.39%
# real_images = [web_images_real_dict, wblots_real_dict, efros_real_dict]
# secondo training: P = 128, N = 5, only image-to-image translation models
# real_images = [pix2pix_real_dict, efros_real_i2i_dict]
# terzo training: P = 128, N = 5, only Efros image-to-image translation models
# real_images = [efros_real_i2i_dict]
# quarto training : P = 128, N = 1, all images in RGB, aug, and then convert to gray
# real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# quinto training : P = 128, N = 4, ONLY STARGAN2 images in RGB, aug, and then convert to gray
# real_images = [stargan2_real_dict]
# sesto training : P = 128, N = 4, ONLY CYCLEGAN images in RGB, aug, and then convert to gray
# real_images = [efros_real_i2i_cyclegan_dict]
# settimo training : P = 128, N = 1, all images in RGB, aug, convert to gray, normalize every img by its mean/std
# real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# ottavo training : P = 128, N = 1, all images in RGB, aug
#real_images = [pix2pix_real_dict, web_images_real_dict, efros_real_dict, wblots_real_dict, stargan2_real_dict]
# test sulle immagini nvidia con ffhq:
# real_images = [ffhq_real_dict]
# test sulle immagini nvidia con le pristine dell'evaluation 1:
#real_images = [eval1_pristine_dict]
real_images = [w_blots_real_dict]
tampered_images = [w_blots_tampered_dict]
m3dsynth_images = [m3dsynth_dict]

#################################################### GAN root folders

# PIX2PIX
pix2pix_root = '/nas/public/exchange/semafor/eval1/pix2pix/synth_data/'
pix2pix_keys = next(os.walk(pix2pix_root))[1]
pix2pix_dirs = [os.path.join(pix2pix_root, path) for path in pix2pix_keys]
pix2pix_dict = {pix2pix_keys[i]: pix2pix_dirs[i] for i in range(len(pix2pix_keys))}

# STYLEGAN2
sg2_root = '/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/'
atom = 'stylegan2-config-f-psi-0.5'
sg2_dirs = directory_find(atom, sg2_root)
sg2_dirs.remove(os.path.join('/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/car-512x384', atom))
sg2_keys = ['-'.join(x.split(sg2_root)[-1].split('/')) for x in sg2_dirs]
atom = 'stylegan2-config-f-psi-1.0'
other_dirs = directory_find(atom, sg2_root)
other_dirs.remove(os.path.join('/nas/public/exchange/semafor/eval1/stylegan2/100k-generated-images/car-512x384', atom))
sg2_dirs.extend(other_dirs)
sg2_keys.extend(['-'.join(x.split(sg2_root)[-1].split('/')) for x in other_dirs])
sg2_dict = {sg2_keys[i]: sg2_dirs[i] for i in range(len(sg2_keys))}

# EFROS
atom = '1_fake'
efros_fake_dirs = directory_find(atom, efros_test_root)
# build dictionary
efros_fake_keys = ['-'.join(x.split(efros_test_root)[-1].split(atom)[0].split('/')[:-1]) for x in efros_fake_dirs]
efros_fake_dict = {efros_fake_keys[i]: efros_fake_dirs[i] for i in range(len(efros_fake_keys))}

# EFROS reduced to image-to-image translation models
i2i_models = ['crn', 'cyclegan', 'gaugan', 'imle', 'stargan']

efros_fake_i2i_keys = []
efros_fake_i2i_dirs = []
for model in i2i_models:
    for k_idx, k in enumerate(efros_fake_keys):
        if model in k:
            efros_fake_i2i_keys.append(k)
            efros_fake_i2i_dirs.append(efros_fake_dirs[k_idx])
efros_fake_i2i_dict = {efros_fake_i2i_keys[i]: efros_fake_i2i_dirs[i] for i in range(len(efros_fake_i2i_keys))}
efros_fake_i2i_cyclegan_dict = {efros_fake_i2i_keys[i]: efros_fake_i2i_dirs[i] for i in range(len(efros_fake_i2i_keys))
                                if 'cyclegan' in efros_fake_i2i_keys[i]}

# WBLOTS GENERATED WITH STG2-pytorch
wblots_fake_dict = {'sg2-pytorch': '/nas/home/smandelli/Pycharm_projects/stylegan2-ada-pytorch/out_wblots_trunc-0.7-'
                                   'snapshot-003400'}

# WEB IMAGES GENERATED WITH STG2-tensorflow
web_images_fake_root = '/nas/home/smandelli/Pycharm_projects/stylegan2-tf/results_sg2_web_images/'
atom = '00000-generate-images'
web_images_fake_dirs = directory_find(atom, web_images_fake_root)
web_images_fake_keys = ['-'.join(x.split(web_images_fake_root)[-1].split(atom)[0].split('/')[:-1]) for x in
                        web_images_fake_dirs]
web_images_fake_dict = {web_images_fake_keys[i]: web_images_fake_dirs[i] for i in range(len(web_images_fake_keys))}

# SG2 IMAGES OF MILITARY VEHICLES
sg2_military_vehicles_root = '/nas/home/smandelli/Pycharm_projects/semafor_web_images/synth_vs_real_detection/' \
                             'stylegan-ada-military-vehicles-0309-100'
sg2_military_vehicles_dict = {'military_vehicles': sg2_military_vehicles_root}

# stargan2 fake:
stargan2_fake_afhq_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/expr/results/afhq_N-70'
stargan2_fake_celeba_hq_root = '/nas/home/smandelli/Pycharm_projects/stargan-v2/expr/results/celeba_hq_N-71'
stargan2_fake_dict = {'afhq_N-70': stargan2_fake_afhq_root, 'celeba_hq_N-71':stargan2_fake_celeba_hq_root}

# primo training: P = 128, N = 1 dà TPR@0.1 = 54.39%
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict]
# secondo training: P = 128, N = 5, only image-to-image translation model
# fake_images = [pix2pix_dict, efros_fake_i2i_dict]
# terzo training: P = 128, N = 5, only Efros image-to-image translation models
# fake_images = [efros_fake_i2i_dict]
# quarto training : P = 128, N = 1, all images in RGB, aug, and then convert to gray
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
#                wblots_fake_dict, stargan2_fake_dict]
# quinto training : P = 128, N = 4, ONLY STARGAN2 images in RGB, aug, and then convert to gray
# fake_images = [stargan2_fake_dict]
# sesto training: P = 128, N = 4, ONLY cyclegan images in RGB, aug, and then convert to gray
# fake_images = [efros_fake_i2i_cyclegan_dict]
# settimo training : P = 128, N = 1, all images in RGB, aug, normalize every img by its mean/std
# fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
#                wblots_fake_dict, stargan2_fake_dict]
# ottavo training : P = 128, N = 1, all images in RGB, aug
#fake_images = [pix2pix_dict, sg2_dict, efros_fake_dict, web_images_fake_dict, sg2_military_vehicles_dict,
#               wblots_fake_dict, stargan2_fake_dict]
fake_images = [w_blots_synthetic_dict]

################### FOLDERS
log_dir = 'runs_tampered_img_32'
models_dir = 'weights_tampered_img_32'
results_dir = 'results_realistically_tampered_img'

