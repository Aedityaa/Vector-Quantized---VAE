import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp")


class PairImages(Dataset):
    """Paired (corrupted, clean) tensors aligned by filename."""

    def __init__(self, clean_path, corrupted_path, transform=None, image_size=(128, 128)):
        self.clean_path = clean_path
        self.corrupted_path = corrupted_path
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        clean_names = {
            f for f in os.listdir(clean_path) if f.lower().endswith(_IMAGE_EXT)
        }
        corrupt_names = {
            f for f in os.listdir(corrupted_path) if f.lower().endswith(_IMAGE_EXT)
        }
        self.filenames = sorted(clean_names & corrupt_names)
        if not self.filenames:
            raise ValueError(
                f"No matching image pairs in {clean_path} and {corrupted_path}"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        clean_image = Image.open(os.path.join(self.clean_path, name)).convert("RGB")
        degraded_image = Image.open(os.path.join(self.corrupted_path, name)).convert("RGB")
        clean = self.transform(clean_image)
        degraded = self.transform(degraded_image)
        return degraded, clean