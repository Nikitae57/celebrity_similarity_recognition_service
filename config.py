from enum import Enum

RESNET_MODEL_NAME = 'resnet50'
VGG_MODEL_NAME = 'vgg16'

MODEL_NAME = VGG_MODEL_NAME
CELEB_IMAGES_DIR = 'bin/img/celeb/optimized_vgg'
CELEB_VECTORIZED_IMAGES_DIR = 'bin/vectorized_img/celeb/optimized_vgg'
USER_FACE_CACHE_DIR = 'bin/img/user_face_cache/'


class ResponseConverterType(Enum):
    JSON = 0
    PROTO = 1


CELEB_SIMILARITY_CONVERTER_TYPE = ResponseConverterType.JSON
