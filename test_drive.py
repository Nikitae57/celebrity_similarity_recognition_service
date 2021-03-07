from model.vectorise_image.image_vectorizer import ImageVectorizer
from PIL import Image


img = Image.open('img/me/1.jpg')
vectorizer = ImageVectorizer()
vector = vectorizer.vectorize(img)
print(vector)
