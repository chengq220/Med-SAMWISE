import torch.utils.data
import torchvision

import datasets.transforms_video as T
from .endovis import build as build_endovis

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(dataset_file: str, image_set: str, args):
    if dataset_file == 'endovis2017':
        return build_endovis(image_set, args)
    if dataset_file == 'endovis2018':
        return build_endovis(image_set, args)

    raise ValueError(f'dataset {dataset_file} not supported')
