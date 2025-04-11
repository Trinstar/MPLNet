from typing import Type, Optional, Callable, Dict, Union
import os
import random
from collections import defaultdict
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

import mplnet.lib as lib

NoneType = Type[None]

class NWPU_RESISC45Dataset_10class_5way(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform: Optional[Callable[[Image.Image], Tensor]] = None,
        seed: Optional[int] = None,
        num_shots:  Optional[int] = None,
    ) -> NoneType:
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.seed = seed
        self.num_shots = num_shots
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        
        # Load all data
        abs_path_dirname = os.path.dirname(os.path.abspath(__file__))
        all_data = lib.load_json(os.path.join(abs_path_dirname, 'files', 'nwpu_10class.json'))
        
        # Group by category
        category_dict = defaultdict(list)
        for item in all_data:
            category = item[2]
            category_dict[category].append(item)
        
        # Select 5 categories based on seed
        random.seed(self.seed)
        selected_categories = random.sample(list(category_dict.keys()), 5)
        print("Selected 5 categories:", selected_categories)
        
        # Split data for each selected category
        train_samples = []
        test_samples = []
        for cat in selected_categories:
            samples = category_dict[cat].copy()
            # Use fixed seed to ensure reproducibility
            random.seed(self.seed)
            random.shuffle(samples)
            train_samples.extend(samples[:self.num_shots])
            test_samples.extend(samples[self.num_shots:])

        # Choose data based on split
        if split == 'train':
            self.samples = train_samples
        elif split == 'test':
            self.samples = test_samples
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        # Build paths, names, and labels
        self.paths = []
        self.targets = []
        self.names = []
        for sample in self.samples:
            self.paths.append(os.path.join(self.root, 'image', sample[0]))
            self.targets.append(sample[1])
            self.names.append(sample[2])
        
        # Label mapping
        self.selected_categories = selected_categories
        self.category_to_label = {cat: idx for idx, cat in enumerate(selected_categories)}
        self.labels = [self.category_to_label[name] for name in self.names]
        
       

        # Other attributes
        self.all_names = selected_categories
        self.num_classes = str(len(self.all_names))
        self.level = 'nwpu_resisc45_10class_5way'

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str, int]]:
        path = self.paths[index]
        target = self.labels[index]
        img = Image.open(path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return {
            "image": img,
            "target": target,
            "path": path,
            "index": index,
            "name": self.names[index],
        }