import os

import pydicom
from skimage import transform, io


def _file_paths(dir):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file():
            if entry.path.endswith('.IMA') or '/DICOM/' in entry.path:
                paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path))
    return paths


if __name__ == '__main__':
    target_dir = '../../../Datasets/Dicom512/'
    root = '../../../Datasets/体检胸片/'
    paths = _file_paths(root)

    for path in paths:
        dc = pydicom.dcmread(path)
        if 'PixelData' in dc:
            im = dc.pixel_array / dc.pixel_array.max()
            if dc.Modality != 'CT' and dc.Modality != 'DX' and dc.pixel_array[0][0] + dc.pixel_array[0][-1] + dc.pixel_array[-1][-1] + dc.pixel_array[-1][0] > 2.5:
                im = 1.0 - im
            im = transform.resize(im, [512, 512])
            io.imsave(target_dir + dc.PatientID + '_' + dc.Modality + '.png', im, as_gray=True)
