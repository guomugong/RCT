import numpy as np
import torch
import cv2
from tqdm import tqdm
import torch.nn as nn
from model.unet_model import UNet
from utils.dataset import FundusSeg_Loader
from utils.eval_metrics import perform_metrics,cal_f1
import copy
import sys
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_domain=sys.argv[1]
target_domain=sys.argv[2]
run_spe=sys.argv[3]

print(f'source domain: {source_domain}')
print(f'target domain: {target_domain}')
#model_path='./snapshot/'+source_domain+'_b'+run_spe+'.pth'
model_path='./snapshot/pretrain_drive.pth'

if source_domain == "drive":
    dataset_mean=[0.4969, 0.2702, 0.1620]
    dataset_std=[0.3479,0.1896,0.1075]

if source_domain == "stare":
    dataset_mean=[0.5889, 0.3272, 0.1074]
    dataset_std=[0.3458,0.1844,0.1104]

if source_domain == "chase":
    dataset_mean=[0.4416, 0.1606, 0.0277]
    dataset_std=[0.3530,0.1407,0.0366]


if target_domain == "drive":
    test_data_path = "./dataset/drive/test/"

if target_domain == "chase":
    test_data_path = "./dataset/chase_db1/test/"

if target_domain == "stare":
    test_data_path = "./dataset/stare/test/"

save_path='./results/'

if __name__ == "__main__":
    with torch.no_grad():
        test_dataset = FundusSeg_Loader(test_data_path,0, target_domain, dataset_mean, dataset_std)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        print('Testing images: %s' %len(test_loader.dataset))
        device = torch.device('cuda')
        net = UNet(n_channels=3, n_classes=1)
        net.to(device=device)
        print(f'Loading model {model_path}')
        net.load_state_dict(torch.load(model_path, map_location=device))

        net.eval()
        pre_stack = []
        label_stack = []

        for image, image2, image3, label, filename, raw_height, raw_width in test_loader:
            image = image.cuda().float()
            label = label.cuda().float()

            image = image.to(device=device, dtype=torch.float32)
            pred = net(image)
            # Normalize to [0, 1]
            pred = torch.sigmoid(pred)
            pred  = pred[:,:,:raw_height,:raw_width]  
            label = label[:,:,:raw_height,:raw_width]
            pred = pred.cpu().numpy().astype(np.double)[0][0]  
            label = label.cpu().numpy().astype(np.double)[0][0]

            pre_stack.append(pred)
            label_stack.append(label)

            pred = pred * 255
            save_filename = save_path + filename[0] + '.png'
            cv2.imwrite(save_filename, pred)
            #print(f'{save_filename} done!')

        print('Evaluating...')
        label_stack = np.stack(label_stack, axis=0)
        pre_stack = np.stack(pre_stack, axis=0)
        label_stack = label_stack.reshape(-1)
        pre_stack = pre_stack.reshape(-1)

        if target_domain == "rimone" or target_domain == 'refuge' or target_domain == "idrid":
            f1 = cal_f1(pre_stack, label_stack)
            print(f'F1-score: {f1}')
        else:
            precision, sen, spec, f1, acc, roc_auc, pr_auc = perform_metrics(pre_stack, label_stack)
            print(f'Precision: {precision} Sen: {sen} Spec:{spec} F1-score: {f1} Acc: {acc} ROC_AUC: {roc_auc} PR_AUC: {pr_auc}')
