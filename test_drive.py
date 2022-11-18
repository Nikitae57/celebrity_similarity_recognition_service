from model.celeb_similarity.most_similar_face.image_vectorizer import ImageVectorizer
from PIL import Image
import numpy as np

# img = Image.open('img/me/1.jpg')
# vectorizer = ImageVectorizer()
# vector = vectorizer.vectorize(img)
# print(vector)
# from model.celeb_similarity.similarity_predictor.predictor import Predictor
#
# predictor = Predictor()
# predictor.predict(123)

features = np.load('dev_files/rcmalli_vggface_labels_v2.npy')
with open('features.txt', 'w') as f:
    for feature in features:
        feature = feature.replace('')