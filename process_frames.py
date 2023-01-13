import os, sys
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
from PIL import Image
import datasets.transforms as T
from tqdm import tqdm


model_config_path = "config/DINO/DINO_4scale_swin_archery.py" # change the path of the model config file
model_checkpoint_path = "logs/archery/checkpoint_best_regular.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
_ = model.eval()

# load coco names
with open('COCO_archery/annotations/instances_train2017.json') as f:
    cats = json.load(f)
    cats = cats['categories']
    id2name = {cat['id']:cat['name'] for cat in cats}
    id2color = {cat['id']:cat['color'] for cat in cats}

os.makedirs('boxed_frames', exist_ok=True)
images = sorted(os.listdir('all_frames/'))
thershold = 0.5 # set a thershold
vslzr = COCOVisualizer()

for image_name in tqdm(images):
    image = Image.open("all_frames/" + image_name).convert("RGB") # load image

    transform = T.Compose([
        #T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(image, None)

    # predict images
    with torch.no_grad():
        output = model.cuda()(image[None].cuda())
        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

    scores = output['scores']
    labels = output['labels']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold

    box_label = [id2name[int(item)] for item in labels[select_mask]]
    box_color = [id2color[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': torch.Tensor([image.shape[1], image.shape[2]]),
        'box_label': box_label,
        'box_color': box_color,
        'image_name': image_name
    }
    vslzr.visualize_cv(image, pred_dict, savedir='./boxed_frames/')