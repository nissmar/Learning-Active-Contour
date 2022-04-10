from load import load_nii
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob
import torch

data_transforms = [
    transforms.Resize((256, 256))
]


class MedicalDataset(Dataset):
    def __init__(self, target=1, root="train", transform=data_transforms):
        ''' target: 1, 2 or 3'''
        im_files = sorted(glob(root + "/*/*[!_gt][!d].nii.gz"))
        gt_files = sorted(glob(root + "/*/*_gt.nii.gz"))
        self.images = []
        self.ground_truths = []
        for im, gt in zip(im_files, gt_files):
            ims = load_nii(im)[0]
            gts = load_nii(gt)[0]
            self.images += [ims[:, :, i] for i in range(ims.shape[2])]
            self.ground_truths += [(gts[:, :, i] == target)
                                   for i in range(gts.shape[2])]
        self.transform = transforms.Compose(transform)

    def __getitem__(self, index):
        img = torch.tensor(self.images[index] /
                           255.0, dtype=torch.float32)[None, :]
        gt = torch.tensor(self.ground_truths[index], dtype=torch.float32)[
            None, :]
        return self.transform(img), self.transform(gt)

    def __len__(self):
        return len(self.images)
