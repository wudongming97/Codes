import numpy as np

from datas import utils, coco
import skimage.io as io

dataset = coco.CocoDataset()
dataset.load_coco('./datasets/coco/', 'train', auto_download=True, class_ids=[18])  # 17 cat 18 dog
dataset.prepare()

print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
# image_ids = np.random.choice(dataset.image_ids, 1)
for image_id in dataset.image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    mask = mask.astype(np.int)
    # visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
    image, mask = utils.resize_image_with_mask(image, mask, 280)
    # visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
    image, mask = utils.crop_image_centered_mask(image, mask, 256)
    # visualize.display_top_masks(image, mask, class_ids, dataset.class_names, limit=1)
    io.imsave('./datasets/dogs_with_mask/{}_cat.png'.format(image_id), image / 255)
    mask = mask[:, :, 0]  # 只取一个
    io.imsave('./datasets/dogs_with_mask/{}_m.png'.format(image_id), np.squeeze(mask))
