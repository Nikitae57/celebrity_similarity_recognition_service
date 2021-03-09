import os
import traceback
from flask import Flask, redirect, request, abort
from werkzeug.exceptions import HTTPException

import config
from config import ResponseConverterType
from controller.celeb_similarity.controller import Controller
from model.celeb_similarity.predictions_converter.json_converter import JsonConverter
from model.celeb_similarity.predictions_converter.proto_converter import ProtoConverter

app = Flask(__name__)
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def init_celeb_similarity_controller():
    if config.CELEB_SIMILARITY_CONVERTER_TYPE == ResponseConverterType.JSON:
        response_converter = JsonConverter()
    elif config.CELEB_SIMILARITY_CONVERTER_TYPE == ResponseConverterType.PROTO:
        response_converter = ProtoConverter()
    else:
        raise ValueError('Unknown celeb similarity controller type')

    return Controller(
        model_name=config.MODEL_NAME,
        faces_img_dir=config.CELEB_IMAGES_DIR,
        faces_vectors_dir=config.CELEB_VECTORIZED_IMAGES_DIR,
        user_face_cache_dir=config.USER_FACE_CACHE_DIR,
        response_converter=response_converter
    )


celeb_similarity_controller = init_celeb_similarity_controller()


def is_allowed_file(filename: str) -> bool:
    _, extension = os.path.splitext(filename)
    extension = extension.replace('.', '')

    return extension in ALLOWED_IMAGE_EXTENSIONS


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return redirect(request.url)

        image = request.files['image']

        if image.filename == '':
            return redirect(request.url)

        if image and is_allowed_file(image.filename):
            return celeb_similarity_controller.process_image(image)

        return abort(400, {'message': f'Invalid file extension. Supported extensions: {ALLOWED_IMAGE_EXTENSIONS}'})
    except HTTPException:
        raise
    except:
        traceback.print_exc()
        abort(503)

    return ''


# @app.route('/api/v1/cropped_face/{}', methods=['GET'])
# def get_cropp():
#     try:
#
#         return abort(400, {'message': f'Invalid file extension. Supported extensions: {ALLOWED_IMAGE_EXTENSIONS}'})
#     except HTTPException:
#         raise
#     except:
#         traceback.print_exc()
#         abort(503)
#
#     return ''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True, use_reloader=False)
