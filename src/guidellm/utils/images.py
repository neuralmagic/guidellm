from io import BytesIO
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger
from PIL import Image
from pydantic import ConfigDict, Field, computed_field

from guidellm.config import settings
from guidellm.core.serializable import Serializable

__all__ = ["load_images", "ImageDescriptor"]

class ImageDescriptor(Serializable):
    """
    A class to represent image data in serializable format.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: Optional[str] = Field(description="url address for image.")
    image: Image.Image = Field(description="PIL image", exclude=True)
    filename: Optional[int] = Field(
        default=None,
        description="Image filename.",
    )

    @computed_field # type: ignore[misc]
    @property
    def image_resolution(self) -> List[int]:
        if self.images is None:
            return None
        else:
            return [im.size for im in self.images]


def load_images(data: str, image_resolution: Optional[List[int]]) -> List[ImageDescriptor]:
    """
    Load an HTML file from a path or URL

    :param data: the path or URL to load the HTML file from
    :type data: Union[str, Path]
    :return: Descriptor containing image url and the data in PIL.Image.Image format
    :rtype: ImageDescriptor
    """

    images = []
    if not data:
        return None
    if isinstance(data, str) and data.startswith("http"):
        response = requests.get(data, timeout=settings.request_timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for img_tag in soup.find_all("img"):
            img_url = img_tag.get("src")

            if img_url:
                # Handle relative URLs
                img_url = urljoin(data, img_url)

                # Download the image
                logger.debug("Loading image: {}", img_url)
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                image = Image.open(BytesIO(img_response.content))

                if image_resolution is not None:
                    image = image.resize(image_resolution)

                # Load image into Pillow
                images.append(
                    ImageDescriptor(
                        url=img_url,
                        image=Image.open(BytesIO(img_response.content)),
                    )
                )

        return images
