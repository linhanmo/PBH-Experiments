import json
import os

def clean_dataset(image_dir, val_json_paths, image_key='file_name'):
    keep_images = set()
    for val_json in val_json_paths:
        if os.path.exists(val_json):
            with open(val_json, 'r') as f:
                data = json.load(f)
                if 'images' in data:
                    for img in data['images']:
                        keep_images.add(img[image_key])
                elif isinstance(data, list): # mpii_val might be a list?
                    for item in data:
                        if 'image' in item:
                            keep_images.add(item['image'])
                        elif 'file_name' in item:
                            keep_images.add(item['file_name'])

    if not keep_images:
        print(f"No images found to keep for {image_dir}. Skipping.")
        return

    print(f"Total images to keep in {image_dir}: {len(keep_images)}")
    
    deleted = 0
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.endswith(('.jpg', '.png', '.jpeg')):
                if f not in keep_images:
                    os.remove(os.path.join(image_dir, f))
                    deleted += 1
    print(f"Deleted {deleted} unused images from {image_dir}")

# AP-10K
clean_dataset(
    '/home/lin/PoseBH/dataset/ap-10k/data',
    ['/home/lin/PoseBH/dataset/ap-10k/annotations/ap10k-val-split1.json',
     '/home/lin/PoseBH/dataset/ap-10k/annotations/ap10k-val-split2.json',
     '/home/lin/PoseBH/dataset/ap-10k/annotations/ap10k-val-split3.json']
)

# OCHuman
clean_dataset(
    '/home/lin/PoseBH/preprocess/ochuman/images',
    ['/home/lin/PoseBH/preprocess/ochuman/annotations/val.json', '/home/lin/PoseBH/dataset/OCHuman/ochuman.json'] # wait, ochuman.json has val?
)

# MPII
clean_dataset(
    '/home/lin/PoseBH/preprocess/mpii/images',
    ['/home/lin/PoseBH/preprocess/mpii/annotations/mpii_val.json']
)
