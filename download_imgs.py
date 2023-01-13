from superannotate import SAClient
from tqdm import tqdm


sa = SAClient()

imgs = sa.search_items('Image set', recursive=True, annotation_status='Completed')
for img_meta in tqdm(imgs):
    sa.download_image(img_meta['path'], img_meta['name'], './images')