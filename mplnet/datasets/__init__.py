from typing import Callable, Tuple, Dict,  Optional
import os

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from mplnet.datasets.nwpu_resisc45 import NWPU_RESISC45Dataset
from mplnet.datasets.rsd46 import RSD46Dataset
from mplnet.datasets.aid import AIDDataset
from mplnet.datasets.nwpu_resisc45_10class_5way import NWPU_RESISC45Dataset_10class_5way

from mplnet.datasets.templates import (
    get_templates,
    BASE_PROMPT,
    IMAGENET_TEMPLATES_SELECT,
    IMAGENET_TEMPLATES,
    RandomTemplate,
    CALTECH_101_TEMPLATE,
    OXFORD_PETS_TEMPLATE,
    STANFORD_CARS_TEMPLATE,
    FLOWERS_102_TEMPLATE,
    FOOD_101_TEMPLATE,
    AIRCRAFT_TEMPLATE,
    SUN_397_TEMPLATE,
    DTD_TEMPLATE,
    EUROSAT_TEMPLATE,
    UCF_101_TEMPLATE,
)

import mplnet.datasets.tools as tools


__all__ = [
    "Caltech101Dataset",
    "DTDDataset",
    "EuroSATDataset",
    "FGVCAircraftDataset",
    "Flowers102Dataset",
    "Food101Dataset",
    "ImageFolder",
    "ImageList",
    "ImagenetADataset",
    "ImagenetRDataset",
    "ImagenetSketchDataset",
    "ImagenetDataset",
    "ImagenetV2Dataset",
    "OxfordPetsDataset",
    "StanfordCarsDataset",
    "SUN397Dataset",
    "UCF101Dataset",
    "NWPU_RESISC45Dataset",
    "RSD46Dataset"
    "AID"
    "NWPU_RESISC45_10class_5wayDataset"

    "get_templates",
    "BASE_PROMPT",
    "IMAGENET_TEMPLATES_SELECT",
    "IMAGENET_TEMPLATES",
    "RandomTemplate",
    "CALTECH_101_TEMPLATE",
    "OXFORD_PETS_TEMPLATE",
    "STANFORD_CARS_TEMPLATE",
    "FLOWERS_102_TEMPLATE",
    "FOOD_101_TEMPLATE",
    "AIRCRAFT_TEMPLATE",
    "SUN_397_TEMPLATE",
    "DTD_TEMPLATE",
    "EUROSAT_TEMPLATE",
    "UCF_101_TEMPLATE",

    "tools",
]


def return_train_val_datasets(
    name: str,
    data_dir: str,
    train_transform: Callable[[Image.Image], Tensor],
    val_transform: Callable[[Image.Image], Tensor],
    seed: Optional[int] = None,
    num_shots: Optional[int] = None,
) -> Tuple[Dataset, Dataset, str]:
    if name == 'nwpu_resisc45':
        train_dataset = NWPU_RESISC45Dataset(
            root=os.path.join(data_dir, 'nwpu_resisc45'),
            split='train',
            transform=train_transform,
        )
        val_dataset = NWPU_RESISC45Dataset(
            root=os.path.join(data_dir, 'nwpu_resisc45'),
            split='test',
            transform=val_transform,
        )
        template = EUROSAT_TEMPLATE

    elif name == 'nwpu_resisc45_10class_5way':
        train_dataset = NWPU_RESISC45Dataset_10class_5way(
        root=os.path.join(data_dir, 'nwpu_resisc45'),
        split='train',
        transform=train_transform,
        seed=seed,
        num_shots = num_shots
        )
        val_dataset = NWPU_RESISC45Dataset_10class_5way(
        root=os.path.join(data_dir, 'nwpu_resisc45'),
        split='test',
        transform=train_transform,
        seed=seed,
        num_shots = num_shots
    )
        template = EUROSAT_TEMPLATE

    elif name == 'aid':
        train_dataset = AIDDataset(
            root=os.path.join(data_dir, 'aid'),
            split='train',
            transform=train_transform,
        )
        val_dataset = AIDDataset(
            root=os.path.join(data_dir, 'aid'),
            split='test',
            transform=val_transform,
        )
        template = EUROSAT_TEMPLATE
    elif name == 'rsd46':
        train_dataset = RSD46Dataset(
            root=os.path.join(data_dir, 'rsd46'),
            split='train',
            transform=train_transform,
        )
        val_dataset = RSD46Dataset(
            root=os.path.join(data_dir, 'rsd46'),
            split='test',
            transform=val_transform,
        )
        template = EUROSAT_TEMPLATE

    else:
        raise NotImplementedError(f"Dataset {name} not implemented")

    return train_dataset, val_dataset, template


def return_ood_loaders(
    data_dir: str,
    transform: Callable[[Image.Image], Tensor],
) -> Dict[str, DataLoader]:
    return {}

def return_domains_loaders(
    data_dir: str,
    transform: Callable[[Image.Image], Tensor],
) -> Dict[str, DataLoader]:
    return {}
