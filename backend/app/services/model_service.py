from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


ROOT = Path(__file__).resolve().parents[3]

CHECKPOINT_PATH = (
    ROOT
    / "models"
    / "checkpoints"
    / "colormate_resnet18_experiment01.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


@lru_cache(maxsize=1)
def load_model():
    """Load the best ResNet18 checkpoint once."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    if len(class_names) != num_classes:
        raise ValueError(
            "Checkpoint class names do not match num_classes."
        )

    model = resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, class_names, checkpoint


def predict_image(image_bytes: bytes) -> dict:
    """Run ResNet18 inference on an uploaded image."""

    model, class_names, checkpoint = load_model()

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_index = probabilities.max(dim=1)

    class_index = int(predicted_index.item())
    predicted_class = class_names[class_index]

    return {
        "predicted_class": predicted_class,
        "season": predicted_class.split("_")[0],
        "confidence_score": float(confidence.item()),
        "class_index": class_index,
        "model": "ResNet18",
        "experiment": checkpoint["experiment"],
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(DEVICE),
    }