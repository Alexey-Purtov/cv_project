# from ultralytics import YOLO
# from PIL import Image, ImageFilter
# import numpy as np

# def blur_faces(model_path: str, images):
#     model = YOLO(model_path)

#     blurred_images = []

#     for img in images:
#         np_img = np.array(img.convert("RGB"))

        
#         results = model.predict(source=np_img, save=False, conf=0.3)[0]

#         for box in results.boxes.xyxy.cpu().numpy().astype(int):
#             x1, y1, x2, y2 = box
#             x1, y1 = max(x1, 0), max(y1, 0)
#             x2, y2 = min(x2, np_img.shape[1]), min(y2, np_img.shape[0])

#             pil_img = img.copy()
#             face_crop = pil_img.crop((x1, y1, x2, y2))
#             blurred_crop = face_crop.filter(ImageFilter.GaussianBlur(15))
#             pil_img.paste(blurred_crop, (x1, y1))
        
#         blurred_images.append(pil_img)

#     return blurred_images

from ultralytics import YOLO
from PIL import Image, ImageFilter
import numpy as np

def blur_faces(model_path: str, images):
    model = YOLO(model_path)
    blurred_images = []

    for img in images:
        np_img = np.array(img.convert("RGB"))
        result = model.predict(source=np_img, save=False, conf=0.05)[0]  

        print("Найдено боксов:", len(result.boxes))

        pil_img = img.copy()
        for box in result.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, np_img.shape[1]), min(y2, np_img.shape[0])

            face_crop = pil_img.crop((x1, y1, x2, y2))
            blurred_crop = face_crop.filter(ImageFilter.GaussianBlur(15))
            pil_img.paste(blurred_crop, (x1, y1))

        blurred_images.append(pil_img)

    return blurred_images