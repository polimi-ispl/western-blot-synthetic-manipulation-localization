"""
Test the patches extracted from images
"""

# Libraries import #
import os
import argparse
from collections import OrderedDict
import gc
import pandas as pd
import torch
import numpy as np
from typing import List
torch.manual_seed(21)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.multiprocessing import cpu_count
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import albumentations as A
import albumentations.pytorch as Ap
from utils import architectures, data_loader
from utils.utils import make_train_tag
from utils.params import models_dir, results_dir
from sklearn.metrics import roc_curve, roc_auc_score
from python_patch_extractor import PatchExtractor as PE


def get_augmentation(aug_list: List, aug_prob: float, jpeg_aug_prob: float, patch_size: int) -> List:
    transform = []
    if 'flip' in aug_list:
        transform.append(A.HorizontalFlip(p=aug_prob))
        transform.append(A.VerticalFlip(p=aug_prob))
    if 'rotate' in aug_list:
        transform.append(A.RandomRotate90(p=aug_prob))
    if 'clahe' in aug_list:
        transform.append(A.CLAHE(p=aug_prob))
    if 'blur' in aug_list:
        transform.append(A.Blur(p=aug_prob))
    if 'crop&resize' in aug_list:
        transform.append(A.RandomSizedCrop((64, patch_size - 1), height=patch_size, width=patch_size, p=aug_prob))
    if 'brightness&contrast' in aug_list:
        transform.append(A.RandomBrightnessContrast(p=aug_prob))
    if 'jitter' in aug_list:
        transform.append(A.ColorJitter(p=aug_prob))
    if 'downscale' in aug_list:
        transform.append(A.Downscale(p=aug_prob))
    if 'hsv' in aug_list:
        transform.append(A.HueSaturationValue(p=aug_prob))
    if 'resize&crop' in aug_list:
        transform.append(A.RandomScale(scale_limit=(1.1, 4), p=aug_prob))
        transform.append(A.RandomCrop(height=patch_size, width=patch_size, always_apply=True, p=1.0))
    if 'jpeg' in aug_list:
        transform.append(A.ImageCompression(quality_lower=40, quality_upper=100, p=jpeg_aug_prob))
        # fisso un particolare fattore di qualità:

    return transform


# Helper functions and classes #

def process_dataset(df: pd.DataFrame,
                    net: architectures.FeatureExtractor,
                    criterion,
                    dataset_class,
                    subsample,
                    num_classes: int,
                    transformer,
                    aug_transformers,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    patch_size: int,
                    patch_number: int,
                    pe_stride: int,
                    ) -> dict:
    if isinstance(device, (int, str)):
        device = torch.device(device)

    dataset = dataset_class(db=df, transformer=transformer, aug_transformers=aug_transformers, subsample=subsample,
                            patch_size=patch_size, patch_number=patch_number, pe_stride=pe_stride)

    # Preallocate
    scores = []
    labels = np.array([])
    preds = np.array([])

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True)
    with torch.no_grad():
        for batch_data in tqdm(loader):

            batch_images = batch_data[0].view(-1, 3, patch_size, patch_size).to(device)
            # Access scalar value from tensor
            number_of_patches = batch_data[2].item()
            batch_out = net(batch_images)
            scores.append(batch_out.cpu().numpy())
            preds = np.append(preds, batch_out.argmax(dim=1).cpu().numpy())

    out_dict = {'scores': scores, 'preds': preds, 'number_of_patches': number_of_patches}
    return out_dict


# Main script #

def main():
    parser = argparse.ArgumentParser()
    # Alternative 1: construct the model path from training hyperparameters
    parser.add_argument('--model', help='Model name', type=str, default='EfficientNetB0')
    parser.add_argument('--db', help='Database name', type=str, default='BasicLoader')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, help='Batch size for backprop in training', default=250)
    parser.add_argument('--aug', help='Augmentation to perform with probability `aug_prob`', nargs='+',
                        default=['flip', 'rotate', 'clahe', 'blur', 'crop&resize', 'brightness&contrast', 'jitter',
                                 'downscale', 'hsv', 'resize&crop', 'jpeg'])
    parser.add_argument('--aug_prob', help='Probability of augmentation', type=float, default=0.5)
    parser.add_argument('--jpeg_aug_prob', help='Probability of JPEG augmentation', type=float, default=0.8)
    parser.add_argument('--patch_size', type=int, default=128, help='P where PxP is the dimension of the patch')
    parser.add_argument('--patch_number', help='N, number of patches per image to extract', type=int, default=1)
    parser.add_argument('--test_patch_number', help='test_N, number of patches per TEST image to extract', type=int,
                        default=1)
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes to consider for training')
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default=models_dir)
    # Alternative 2: pass as argument the complete model path directly
    parser.add_argument('--test_model_path', action='store', help='Path to the model`s weights to test')
    # Common params
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--workers', type=int, default=cpu_count() // 2)
    parser.add_argument('--test_batch_size', type=int, default=1, help='Batch size used in test')
    parser.add_argument('--test_pkl_file', type=str, help='pkl file to be tested', required=True)
    parser.add_argument('--subsample', type=float, help='Fraction to subsample datasets')
    parser.add_argument('--results_dir', type=str, help='Directory for saving the test results',
                        default=results_dir)
    parser.add_argument('--override', action='store_true', help='Override previous test results')
    parser.add_argument('--testaug', help='Augmentation in test to perform with probability 1',
                        nargs='+')
    parser.add_argument('--patch_extractor_stride', type=int, default=4, help='Stride value used when extracting patches')

    args = parser.parse_args()

    # Parse arguments
    gpu = args.gpu
    workers = args.workers
    subsample = args.subsample
    net_name = args.model
    db_name = args.db
    lr = args.lr
    batch_size = args.batch_size
    aug_list = args.aug
    aug_prob = args.aug_prob
    jpeg_aug_prob = args.jpeg_aug_prob
    P = args.patch_size
    N = args.patch_number
    num_classes = args.num_classes
    classes = np.arange(0, num_classes)
    weights_folder = args.models_dir
    results_folder = args.results_dir
    model_path = args.test_model_path
    override = args.override
    test_pkl_file = args.test_pkl_file
    test_batch_size = args.test_batch_size
    testaug_list = args.testaug
    test_N = args.test_patch_number
    pe_stride = args.patch_extractor_stride

    # GPU configuration
    device = torch.device('cuda:{}'.format(gpu)) if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    # Parse network params
    if model_path is None:
        if net_name is None:
            raise RuntimeError('Net name is required if \"model_path\" is not provided')

        network_class = getattr(architectures, net_name)
        train_tag = make_train_tag(network_class, lr, aug_list, aug_prob, P, N, batch_size, num_classes)
        model_path = os.path.join(weights_folder, train_tag, 'bestval.pth')
    else:
        # Parse arguments from model path
        net_name = str(model_path).split('net-')[1].split('_')[0]
        num_classes = 2
        train_tag = str(os.path.split(os.path.split(model_path)[0])[1])

    # Instantiate and load network
    pe = PE.PatchExtractor(dim=(P, P, 3), stride=(pe_stride, pe_stride, 3))
    network_class = getattr(architectures, net_name)
    net = network_class(n_classes=num_classes, pretrained=False).eval().to(device)
    print('Loading model...')
    state_tmp = torch.load(model_path, map_location='cpu')
    if 'net' not in state_tmp.keys():
        state = OrderedDict({'net': OrderedDict()})
        [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
    else:
        state = state_tmp
    incomp_keys = net.load_state_dict(state['net'], strict=True)
    print(incomp_keys)
    print('Model loaded!')

    # Transformer and augmentation
    net_normalizer = net.get_normalizer()
    transform = [
        A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std, ),
        Ap.transforms.ToTensorV2(),
    ]

    aug_transformers = get_augmentation(testaug_list, 1, 1, P) if testaug_list else None

    # Loss-measure
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # Initialize folder for results
    results_folder = os.path.join(results_folder, train_tag)
    os.makedirs(results_folder, exist_ok=True)

    # Instantiate Dataset class for test
    dataset_class = getattr(data_loader, db_name)

    # Instantiate DataFrames for tests
    processing_list = []
    df = pd.read_pickle('{}'.format(test_pkl_file))
    df = df[:15]


    if testaug_list is not None:
        aug_path = 'testaug'
        for test_aug in testaug_list:
            aug_path = aug_path + '-' + test_aug
        results_folder = os.path.join(results_folder, aug_path)
        os.makedirs(results_folder, exist_ok=True)

    pkl_dataset_name = test_pkl_file.split('.pkl')[0].split('/')[-1]
    processing_list.append((df, os.path.join(results_folder, 'test_s{}_{}_img_results.pkl'.
                                             format(pe.stride[0], pkl_dataset_name)), pkl_dataset_name))

    # Testing loop
    labels = np.array([])
    preds = np.array([])
    for df, out_df_path, set_tag in processing_list:
        print('Testing {} on {} split...'.format(train_tag, set_tag))
        dataset_out = process_dataset(df, net, criterion, dataset_class, subsample, num_classes,
                                      transform, aug_transformers, test_batch_size, workers, device, P, test_N, pe_stride)
        df['scores'] = [score for score in dataset_out['scores']]
        scores = dataset_out['scores']
        preds = dataset_out['preds']
        number_of_patches = dataset_out['number_of_patches']

        # Reconstruct patches
        heatmap_path_list = []
        heatmap_arr = np.array([])
        mask_arr = np.array([])
        auc_list = []
        ba_max_list = []
        thr_max_list = []
        for img in range(len(scores)):
            patches = []
            patch_num = 0
            while patch_num < number_of_patches**2:
                patch = np.ones((P, P, 1), dtype=np.uint8)
                # Apply softmax on each score
                score = np.exp(scores[img][patch_num]) / sum(np.exp(scores[img][patch_num]))
                synth_score = score[1]
                patch = patch * synth_score
                patches.append(patch)
                patch_num += 1
            pred_patches = np.array(patches).reshape((number_of_patches, number_of_patches, 1, 32, 32, 1))
            # Setting the value since two different PatchExtractor objects are used for extraction and reconstruction
            pe.in_content_cropped_shape = (256, 256, 1)
            # Reconstruct heatmap
            heatmap = pe.reconstruct(pred_patches)


            mask_path = df['mask'].iloc[img]
            mask = np.array(Image.open(mask_path))
            # Consider only original dimensions (top-left corner) of tampered image with inpainting (dall-e)
            if df['tamper_method'].iloc[img] == "dall_e":
                mask = mask[:256, :256]

            # Common threshold for all data
            heatmap_arr = np.append(heatmap_arr, heatmap.flatten())
            mask_arr = np.append(mask_arr, np.round(mask[:, :, 0].flatten()/255))

            # Metrics
            fpr, tpr, thresholds = roc_curve(np.round(mask[:, :, 0].flatten()/255), heatmap.flatten())
            auc = roc_auc_score(np.round(mask[:, :, 0].flatten()/255), heatmap.flatten())
            # AUC minimum value set to 0.5
            if auc < 0.5:
                auc = 1-auc
            ba = (tpr+(1-fpr))/2
            # Drop value in first index, (from the documentation of roc_curve)
            # thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
            ba = ba[1:]
            thresholds = thresholds[1:]
            ba_max = np.max(ba)
            thr_max = thresholds[np.argmax(ba)]
            # Collect metrics
            auc_list.append(auc)
            ba_max_list.append(ba_max)
            thr_max_list.append(thr_max)
            heatmap_path_list.append(heatmap)

        df['auc'] = auc_list
        df['ba_max'] = ba_max_list
        df['thr_max'] = thr_max_list
        df['heatmap'] = heatmap_path_list

        # Compute common threshold
        fpr_comm, tpr_comm, thresholds_comm = roc_curve(mask_arr, heatmap_arr)
        auc_comm = roc_auc_score(mask_arr, heatmap_arr)
        # AUC minimum value set to 0.5
        if auc_comm < 0.5:
            auc_comm = 1 - auc_comm
        ba_comm = (tpr_comm + (1 - fpr_comm)) / 2
        # Drop value in first index, (from the documentation of roc_curve)
        # thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_score) + 1
        ba_comm = ba_comm[1:]
        thresholds_comm = thresholds_comm[1:]
        ba_max_comm = np.max(ba_comm)
        thr_max_comm = thresholds_comm[np.argmax(ba_comm)]
        df['auc_common'] = auc_comm
        df['ba_max_common'] = ba_max_comm
        df['thr_max_comm'] = thr_max_comm


        print('Saving results to: {}'.format(out_df_path))
        # df.to_pickle(out_df_path)

        del (dataset_out)
        del (df)
        gc.collect()

    print('Testing completed! Bye!')


if __name__ == '__main__':
    main()