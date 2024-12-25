import numpy as np
from PIL import Image
import io

def preprocess_image(file:bytes):
    try:
        image = Image.open(io.BytesIO(file))
        print(f"File format: {image.format}")
        if image.format not in ["JPEG", "PNG", "GIF", "BMP", "TIFF", "WEBP"]:
            raise ValueError("Файл не является поддерживаемым растровым изображением.")
        return np.array(image)
    except Exception as e:
        raise ValueError(f"Ошибка обработки файла: {e}")
