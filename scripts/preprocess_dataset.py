import random
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "menes000/armocromia-12-season"

IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_SEED = 42
NUM_WORKERS = 0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================
# REPRODUCIBILITY
# ==========================================

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ==========================================
# PREPROCESSING TRANSFORM
# ==========================================

preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ==========================================
# CUSTOM DATASET
# ==========================================

class ColorMateDataset(Dataset):

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):

        sample = self.dataset[index]

        image = sample["image"].convert("RGB")
        label = int(sample["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ==========================================
# LOAD DATASET
# ==========================================

print("Loading ColorMate dataset...")

dataset = load_dataset(DATASET_NAME)

train_data = ColorMateDataset(
    dataset["train"],
    transform=preprocess_transform
)

test_data = ColorMateDataset(
    dataset["test"],
    transform=preprocess_transform
)


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ==========================================
# VALIDATION
# ==========================================

print("\n" + "=" * 50)
print("COLORMATE PREPROCESSING VALIDATION")
print("=" * 50)

images, labels = next(iter(train_loader))

print("\nTRAIN BATCH")
print("Image shape:", images.shape)
print("Label shape:", labels.shape)
print("Image dtype:", images.dtype)
print("Label dtype:", labels.dtype)
print("Image device:", images.device)
print("Image minimum:", images.min().item())
print("Image maximum:", images.max().item())

assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
assert labels.ndim == 1
assert images.dtype == torch.float32
assert torch.isfinite(images).all()

test_images, test_labels = next(iter(test_loader))

print("\nTEST BATCH")
print("Image shape:", test_images.shape)
print("Label shape:", test_labels.shape)

assert test_images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
assert torch.isfinite(test_images).all()

print("\n" + "=" * 50)
print("PREPROCESSING VALIDATION PASSED")
print("=" * 50)