import os


# from skimage import io
#
# file_name_list = [
#     '0000035767_CR',
#     '0000037845_CR',
#     '0000038462_CR',
#     '0000038701_CR',
#     '0000040051_CR',
#     '0000042262_CR',
#     '0000044366_CR',
#     '0000045790_CR',
#     '0000051689_CR',
#     '0000056729_CR',
#     '0000057278_CR',
#     '0000059639_CR',
#     '0000065389_CR',
#     '0000076963_CR',
#     '0000081014_CR',
#     '0000085585_CR',
#     '0000096454_CR',
#     '0000099892_CR',
#     '0000103585_CR',
#     '0000106174_CR',
#     '0000175729_CR',
#     '0000178352_CR',
#     '0000179636_CR',
#     '0000181186_CR',
#     '0000182583_CR',
#     '0000186293_CR',
#     '0000200560_CR',
#     '0000202338_CR',
#     '0000204931_CR',
#     '0000205930_CR',
#     '0000207072_CR',
#     '0000209465_CR',
#     '0000212874_CR',
#     '0000220462_CR',
#     '0000239363_CR',
#     '0000246485_CR',
#     '0000251084_CR',
#     '0000256553_CR',
#     '0000261256_CR',
#     '0000267235_CR',
#     '0000274641_CR',
#     '0000278554_CR',
#     '0000281496_CR',
#     '0000310062_CR',
#     '0000311216_CR',
#     '0000312588_CR',
#     '0000324182_CR',
#     '0000325563_CR',
#     '0000329518_CR',
#     '0000334334_CR',
#     '0000340039_CR',
#     '0005426825_CR',
#     '0008302901_CR'
# ]
#
# for fname in file_name_list:
#     ffname = 'F:\\数据集\\Dicom512\\' + fname + '.png'
#     im = io.imread(ffname, as_gray=True)
#     im = 1.0 - im / im.max()
#     io.imsave(ffname, im, as_gray=True)

def _file_paths(dir):
    paths = []
    for entry in os.scandir(dir):
        if entry.is_file():
            if entry.path.endswith('.png'):
                paths.append(entry.path)
        elif entry.is_dir():
            paths.extend(_file_paths(entry.path))
    return paths


root = 'F:\\数据集\\胸骨处理后的数据\\Dicom512_train\\'
paths = _file_paths(root)

for pp in paths:
    if 'DX' in pp:
        newpp = pp.replace('DX', 'CR')
        os.rename(pp, newpp)
