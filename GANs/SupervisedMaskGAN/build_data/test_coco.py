import numpy as np
import skimage.io as io

from build_data import utils, coco

_save_path_map = {
    17: '../datasets/cats_with_mask/',
    18: '../datasets/dogs_with_mask/'
}


def generate_images_with_mask(class_id):
    dataset = coco.CocoDataset()
    dataset.load_coco('../datasets/coco/', 'train', auto_download=True, class_ids=[class_id])  # 17 cat 18 dog
    dataset.prepare()

    print("Image Count: {}".format(len(dataset.image_ids)))
    print("Class Count: {}".format(dataset.num_classes))
    for i, info in enumerate(dataset.class_info):
        print("{:3}. {:50}".format(i, info['name']))

    # Load and display random samples
    # image_ids = np.random.choice(dataset.image_ids, 1)
    for image_id in dataset.image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids_ = dataset.load_mask(image_id)
        mask = mask.astype(np.int)
        image, mask = utils.resize_image_with_mask(image, mask, 280)
        image, mask = utils.crop_image_centered_mask(image, mask, 256)
        # visualize.display_top_masks(image, mask, class_ids_, dataset.class_names, limit=1)
        mask = np.squeeze(mask[:, :, 0])  # 只取一个
        if np.sum(mask) > 50 * 50:
            io.imsave(_save_path_map[class_id] + '{}_cat.png'.format(image_id), image / 255)
            io.imsave(_save_path_map[class_id] + '{}_m.png'.format(image_id), mask)


if __name__ == '__main__':
    for id in _save_path_map.keys():
        generate_images_with_mask(id)
