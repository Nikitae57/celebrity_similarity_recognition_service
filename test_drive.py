import numpy as np
import jsonpickle

from matplotlib import pyplot as plt

from model.celeb_similarity.predictor import Predictor
from model.celeb_similarity.image_preprocessor import ImagePreprocessor

preprocessor = ImagePreprocessor()

pixels = plt.imread('img.jpg')
preprocessed_img = preprocessor.preprocess_image(pixels)
plt.imshow(preprocessed_img.reshape(224,224,3).astype(np.uint8))
plt.show()

recognizer = Predictor()
preds = recognizer.predict(preprocessed_img)
preds_json = jsonpickle.encode(preds, unpicklable=False)

print(preds_json)
