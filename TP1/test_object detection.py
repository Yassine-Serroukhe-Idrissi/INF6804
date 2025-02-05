from transformers import pipeline
from PIL import Image

image = Image.open("TP1_data\car_query.jpg").convert("RGB")

detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")

predictions = detector(
    image,
    candidate_labels=["a photo of a car", "a photo of a dog"],
)
print(predictions)
# [{'score': 0.95,
#   'label': 'a photo of a cat',
#   'box': {'xmin': 180, 'ymin': 71, 'xmax': 271, 'ymax': 178}},
#   ...
# ]
