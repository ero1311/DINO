import json
from tqdm import tqdm
import subprocess
import os
from shutil import move

video_file = './archery.mp4'
annots = './archery.json'

with open(annots) as f:
    annot_data = json.load(f)

os.makedirs('frames', exist_ok=True)

folder_counter = 0
for inst in tqdm(annot_data['instances']):
    for param in inst['parameters']:
        param_folder = 'frames/' + str(folder_counter).zfill(6)
        os.makedirs(param_folder)
        subprocess.run(['ffmpeg', '-ss', str(param['start'] / 1000000), '-to', str(param['end'] / 1000000), '-i', video_file, param_folder + '/%6d.png'])
        folder_counter += 1

os.makedirs('all_frames', exist_ok=True)
folders = sorted(os.listdir('frames'))
counter = 0
for folder in folders:
    imgs = sorted(os.listdir('frames/' + folder))
    for img in imgs:
        move('frames/' + folder + '/' + img, 'all_frames/' + str(counter).zfill(6) + '.png')
        counter += 1