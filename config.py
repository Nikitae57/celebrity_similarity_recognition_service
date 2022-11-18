from enum import Enum

RESNET_MODEL_NAME = 'resnet50'
VGG_MODEL_NAME = 'vgg16'

MODEL_NAME = RESNET_MODEL_NAME
CELEB_IMAGES_DIR = 'static/optimized_vgg/'
CELEB_VECTORIZED_IMAGES_DIR = 'static/optimized_vgg_vectorized/'
CROPPED_USER_FACES = 'static/user_face_cache/'


class ResponseConverterType(Enum):
    JSON = 0
    PROTO = 1


CELEB_SIMILARITY_CONVERTER_TYPE = ResponseConverterType.JSON
