import json
from pathlib import Path
from typing import List

import PIL
import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize,\
                                ColorJitter,RandomResizedCrop,RandomHorizontalFlip,\
                                RandomVerticalFlip,RandomPerspective,RandomGrayscale,RandomChoice,RandomRotation,\
                                    RandomAutocontrast
base_path = Path(__file__).absolute().parents[1].absolute()
base_path_cirr=Path('The path of the folder where the CIRR dataset is stored')
base_path_shoes=Path('The path of the folder where the Shoes dataset is stored')
base_path_fiq=Path('The path of the folder where the FashionIQ dataset is stored')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class SquarePad:
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


class TargetPad:
    def __init__(self, target_ratio: float, size: int):
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def squarepad_transform(dim: int):
    return Compose([
        SquarePad(dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def image_aug_transform(target_ratio=1.25, dim=288):
    return Compose([
        TargetPad(target_ratio, dim),
        RandomResizedCrop(dim),
        _convert_image_to_rgb,
        RandomChoice([
            ColorJitter(brightness=0.5),
            ColorJitter(contrast=0.5),
            ColorJitter(saturation=0.5),
            ColorJitter(hue=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ]),
        RandomChoice([
            RandomRotation((0, 0)),
            RandomHorizontalFlip(p=1),
            RandomVerticalFlip(p=1),
            RandomRotation((90, 90)),
            RandomRotation((180, 180)),
            RandomRotation((270, 270)),
            Compose([
                RandomHorizontalFlip(p=1),
                RandomRotation((90, 90)),
            ]),
            Compose([
                RandomHorizontalFlip(p=1),
                RandomRotation((270, 270)),
            ])
        ]),
        RandomAutocontrast(p=0.5),
        RandomPerspective(p=0.5),
        RandomGrayscale(p=0.5),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class FashionIQDataset(Dataset):
    def __init__(self, split: str, dress_types: List[str], mode: str, preprocess: callable):
        self.mode = mode
        self.dress_types = dress_types
        self.split = split

        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")
        if split not in ['test', 'train', 'val']:
            raise ValueError("split should be in ['test', 'train', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess
        self.image_aug_transform=image_aug_transform()

        self.triplets: List[dict] = []
        for dress_type in dress_types:
            with open(base_path_fiq / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json') as f:
                self.triplets.extend(json.load(f))

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(base_path_fiq / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                image_captions = self.triplets[index]['captions']
                reference_name = self.triplets[index]['candidate']

                if self.split == 'train':
                    reference_image_path = base_path_fiq / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                    raw_pil_image=PIL.Image.open(reference_image_path)
                    reference_image = self.preprocess(raw_pil_image)
                    reference_image_aug=self.image_aug_transform(raw_pil_image)
                    target_name = self.triplets[index]['target']
                    target_image_path = base_path_fiq / 'fashionIQ_dataset' / 'images' / f"{target_name}.jpg"
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, image_captions,reference_image_aug

                elif self.split == 'val':
                    target_name = self.triplets[index]['target']
                    return reference_name, target_name, image_captions

                elif self.split == 'test':
                    reference_image_path = base_path_fiq / 'fashionIQ_dataset' / 'images' / f"{reference_name}.jpg"
                    reference_image = self.preprocess(PIL.Image.open(reference_image_path))
                    return reference_name, reference_image, image_captions

            elif self.mode == 'classic':
                image_name = self.image_names[index]
                image_path = base_path_fiq / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
                image = self.preprocess(PIL.Image.open(image_path))
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")
        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.image_names)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")

class ShoesDataset(Dataset):
    def __init__(self,split,mode,preprocess):
        assert split in ['train','val']
        assert mode in ['relative','classic']
        shoes_img_dir=base_path_shoes/'shoes'/'images'
        shoes_anno_dir=base_path_shoes/'shoes'/'annotations'
        self.split=split
        self.mode=mode
        self.preprocess=preprocess
        self.image_aug_transform=image_aug_transform()
        self.shoes_img_dir=shoes_img_dir
        self.triplets=json.loads(open(shoes_anno_dir/f'triplet.{split}.json','r').read())
        self.image_names=json.loads(open(shoes_anno_dir/f'split.{split}.json','r').read())
        print(f"Shoes {split} dataset in {mode} mode initialized")
    def __getitem__(self,index):
        if self.mode=='relative':
            image_captions=self.triplets[index]['RelativeCaption']
            reference_name=self.triplets[index]['ReferenceImageName']
            if self.split=='train':
                reference_image_path=self.shoes_img_dir/reference_name
                raw_pil_image=PIL.Image.open(reference_image_path)
                reference_image=self.preprocess(raw_pil_image)
                reference_image_aug=self.image_aug_transform(raw_pil_image)
                target_name=self.triplets[index]['ImageName']
                target_image_path=self.shoes_img_dir/target_name
                target_image=self.preprocess(PIL.Image.open(target_image_path))
                return reference_image,target_image,image_captions,reference_image_aug
            elif self.split=='val':
                target_name=self.triplets[index]['ImageName']
                return reference_name,target_name,image_captions
        elif self.mode=='classic':
            image_name=self.image_names[index]
            image_path=self.shoes_img_dir/image_name
            image=self.preprocess(PIL.Image.open(image_path))
            return image_name,image
    def __len__(self):
        if self.mode=='relative':
            return len(self.triplets)
        return len(self.image_names) # classic
        
class CIRRDataset(Dataset):
    def __init__(self, split: str, mode: str, preprocess: callable):
        self.preprocess = preprocess
        self.mode = mode
        self.split = split
        self.image_aug_transform=image_aug_transform()

        if split not in ['test1', 'train', 'val']:
            raise ValueError("split should be in ['test1', 'train', 'val']")
        if mode not in ['relative', 'classic']:
            raise ValueError("mode should be in ['relative', 'classic']")

        # get triplets made by (reference_image, target_image, relative caption)
        with open(base_path_cirr / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.{split}.json') as f:
            self.triplets = json.load(f)

        # get a mapping from image name to relative path
        with open(base_path_cirr / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset in {mode} mode initialized")

    def __getitem__(self, index):
        try:
            if self.mode == 'relative':
                group_members = self.triplets[index]['img_set']['members']
                reference_name = self.triplets[index]['reference']
                rel_caption = self.triplets[index]['caption']

                if self.split == 'train':
                    reference_image_path = base_path_cirr / 'cirr_dataset' / self.name_to_relpath[reference_name]
                    raw_pil_image=PIL.Image.open(reference_image_path)
                    reference_image = self.preprocess(raw_pil_image)
                    reference_image_aug=self.image_aug_transform(raw_pil_image)
                    target_hard_name = self.triplets[index]['target_hard']
                    target_image_path = base_path_cirr / 'cirr_dataset' / self.name_to_relpath[target_hard_name]
                    target_image = self.preprocess(PIL.Image.open(target_image_path))
                    return reference_image, target_image, rel_caption,reference_image_aug

                elif self.split == 'val':
                    target_hard_name = self.triplets[index]['target_hard']
                    return reference_name, target_hard_name, rel_caption, group_members

                elif self.split == 'test1':
                    pair_id = self.triplets[index]['pairid']
                    return pair_id, reference_name, rel_caption, group_members

            elif self.mode == 'classic':
                image_name = list(self.name_to_relpath.keys())[index]
                image_path = base_path_cirr / 'cirr_dataset' / self.name_to_relpath[image_name]
                im = PIL.Image.open(image_path)
                image = self.preprocess(im)
                return image_name, image

            else:
                raise ValueError("mode should be in ['relative', 'classic']")

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        if self.mode == 'relative':
            return len(self.triplets)
        elif self.mode == 'classic':
            return len(self.name_to_relpath)
        else:
            raise ValueError("mode should be in ['relative', 'classic']")
