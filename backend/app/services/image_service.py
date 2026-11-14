from io import BytesIO
import warnings

from PIL import (
    Image,
    ImageFile,
    UnidentifiedImageError,
)


MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_IMAGE_PIXELS = 20_000_000  # 20 megapixels

ALLOWED_IMAGE_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
}

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def validate_image(image_bytes: bytes) -> dict:
    """
    Validate an uploaded image using its actual content.
    """

    if not image_bytes:
        raise ValueError("Image file is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise ValueError("Image size must not exceed 5 MB.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format

                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise ValueError(
                        "Only JPEG, PNG, and WebP images are allowed."
                    )

                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError(
                        "Image dimensions exceed the allowed limit."
                    )

                image.verify()

        return {
            "valid": True,
            "format": image_format,
            "width": image.width,
            "height": image.height,
            "size_bytes": len(image_bytes),
        }

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(
            "The uploaded file is not a valid or safe image."
        ) from error