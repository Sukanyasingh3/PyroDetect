import os
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
import matplotlib.image as mpimg
from keras.models import load_model


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


def index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()

    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)

        # Use full system path instead of fss.url(_image)
        path = os.path.join(settings.MEDIA_ROOT, _image)

        # Read the image

        image = mpimg.imread(path)
        if image is None:
            raise ValueError(
                "Failed to read the image. Make sure the file path is correct."
            )
        resized_image = cv2.resize(image, (256, 256))
        normalized_image = resized_image / 256.0
        input_image = np.expand_dims(normalized_image, axis=0)

        model_path = os.path.join(settings.BASE_DIR, "model.h5")
        model = load_model(model_path)
        result = model.predict(input_image)
        # print("Result:", result[0][0])
        Fire = result[0][0] >= 0.5
        # print("Fire:", Fire)
        if Fire:
            prediction = "Fire"
        else:
            prediction = "No Fire"

        # Pass the filename to the template
        filename = _image  # Assuming you want to pass the full filename
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "filename": filename,  # Include the filename in the context
                "image_url": fss.url(_image),
                "prediction": prediction,
            },
        )

    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "index.html",
            {"message": str(e)},
        )
