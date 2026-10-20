"""Phone photo orientation must be applied before the upload cropper runs."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage

from services.image_service import prepare_opponent_image_for_prediction


@pytest.mark.parametrize('orientation', [1, 3, 6, 8])
def test_upload_applies_exif_orientation_and_preserves_bgr(orientation):
    # Unequal dimensions and colored quadrants catch rotated and mirrored pixels.
    image = Image.new('RGB', (120, 80), (200, 20, 30))
    image.paste((10, 180, 50), (60, 0, 120, 40))
    image.paste((30, 40, 220), (0, 40, 60, 80))
    exif = image.getexif()
    exif[274] = orientation
    buffer = BytesIO()
    image.save(buffer, format='JPEG', exif=exif, quality=95)
    payload = buffer.getvalue()
    with Image.open(BytesIO(payload)) as decoded:
        expected = np.array(ImageOps.exif_transpose(decoded).convert('RGB'))[:, :, ::-1]
    upload = FileStorage(stream=BytesIO(payload), filename='phone.jpg', content_type='image/jpeg')
    actual = prepare_opponent_image_for_prediction(upload)['image']
    np.testing.assert_array_equal(actual, expected)
