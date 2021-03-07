import os
import traceback
from flask import Flask, redirect, request, abort
from werkzeug.exceptions import HTTPException

import config
from controller.celeb_similarity.controller import Controller

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
celeb_similarity_controller = Controller(
    model_name=config.MODEL_NAME,
    faces_img_dir=config.CELEB_IMAGES_FOLDER,
    faces_vectors_dir=config.CELEB_VECTORIZED_IMAGES_FOLDER
)


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7777, debug=True, use_reloader=False)
