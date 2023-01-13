import json

data_path = 'COCO_archery/annotations/instances_train2017.json'

with open(data_path) as f:
    data = json.load(f)

cats = data['categories']
cat_idx = [cat['id'] for cat in cats]
cat_remap = dict(zip(cat_idx, range(len(cat_idx))))
with open('remap_cats.json', 'w') as f:
    json.dump(cat_remap, f)

new_annots = []
new_cats = []
for cat in cats:
    cat['id'] = cat_remap[cat['id']]
    new_cats.append(cat)

print(new_cats)

for ann in data['annotations']:
    ann['category_id'] = cat_remap[ann['category_id']]
    new_annots.append(ann)

data['categories'] = new_cats
data['annotations'] = new_annots

with open(data_path, 'w') as f:
    json.dump(data, f)