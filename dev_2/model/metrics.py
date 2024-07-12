# подключение библиотек
import os
from tqdm import tqdm
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from collections import OrderedDict

from PIL import Image 
import torch
from torch import nn
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision.models.feature_extraction import create_feature_extractor
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
import clip
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

from ignite.engine import Engine
from ignite.utils import manual_seed
from ignite.metrics import InceptionScore


@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1


########## определение констант
seed=42
manual_seed(666)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create default evaluator for doctests
def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)


########## парсинг аргументов
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "orig_path", type=str,
    help="path to original images",
)

parser.add_argument(
    "gen_path", type=str,
    help="path to generated images",
)

args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

if orig_path.split('.')[0] == "":
    orig_path = os.path.join(os.getcwd(), orig_path[2:])
if gen_path.split('.')[0] == "":
    gen_path = os.path.join(os.getcwd(), gen_path[2:])

########## чтение данных
count = min([len(os.listdir(orig_path)), len(os.listdir(gen_path))])
img_size = 448
pil_transform = transforms.Compose([transforms.PILToTensor()])
gen_transform = transforms.Compose([transforms.Resize(size=(img_size, img_size))])


test_paths = sorted([os.path.join(orig_path, name) for name in os.listdir(orig_path) 
                     if int(name.split('.')[0]) in range(count)])
test_gen_paths = sorted([os.path.join(gen_path, name) for name in os.listdir(gen_path) 
                         if int(name.split('.')[0]) in range(count)])


all_images = torch.zeros((count, 3, img_size, img_size))
all_brain_recons = torch.zeros((count, 3, img_size, img_size))

for index, (test_path, test_gen_path) in enumerate(zip(test_paths, test_gen_paths)):
    orig_im = Image.open(test_gen_path)
    orig_tensor = pil_transform(orig_im) / 256
    
    gen_tensor = torch.Tensor(np.load(test_path)).permute(2, 0, 1)
    gen_tensor = gen_transform(gen_tensor) / 256
    
    all_images[index] = orig_tensor
    all_brain_recons[index] = gen_tensor


all_images = all_images.to(device)
all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0,1)


########## PIXCORR
preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])

all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

corrsum = 0
for i in tqdm(range(count)):
    corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
corrmean = corrsum / count

pixcorr = corrmean


########## SSIM
preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR), 
])

img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())
print("converted, now calculating ssim...")

ssim_score=[]
for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
    ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))

ssim = np.mean(ssim_score)


########## AlexNet
alex_weights = AlexNet_Weights.IMAGENET1K_V1

alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
alex_model.eval().requires_grad_(False)

# see alex_weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.4')
alexnet2 = np.mean(all_per_correct)

all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, 
                                                          alex_model, preprocess, 'features.11')
alexnet5 = np.mean(all_per_correct)


########## InceptionV3
weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), 
                                           return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images,
                                        inception_model, preprocess, 'avgpool')
        
inception = np.mean(all_per_correct)


########## CLIP
clip_model, preprocess = clip.load("ViT-L/14", device=device)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images,
                                        clip_model.encode_image, preprocess, None) # final layer
clip_ = np.mean(all_per_correct)



########## Efficient Net
weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), 
                                    return_nodes=['avgpool']).to(device)
eff_model.eval().requires_grad_(False)

# see weights.transforms()
preprocess = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gt = eff_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = eff_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()


########## SWAV
swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
swav_model = create_feature_extractor(swav_model, 
                                    return_nodes=['avgpool']).to(device)
swav_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

gt = swav_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = swav_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()

swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()


########## Inception Score
metric = InceptionScore()
metric.attach(default_evaluator, "is")
state = default_evaluator.run([all_brain_recons])

iscore = state.metrics["is"]


########## FID
metric = FrechetInceptionDistance(feature=64)
metric.update((all_images * 256).type(torch.uint8).to('cpu'), real=True)
metric.update((all_brain_recons * 256).type(torch.uint8).to('cpu'), real=False)

fid = metric.compute().item()


########## Cохранение результатов
data = {
    "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV", "IS", "FID"],
    "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav, iscore, fid],
}

df = pd.DataFrame(data)
df.to_csv('output.csv', sep='\t', index=False)