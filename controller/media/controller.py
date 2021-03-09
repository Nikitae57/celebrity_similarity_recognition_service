import os
from typing import Optional


class MediaController:
    def __init__(self, cropped_faces_dir: str):
        self._preprocess_cropped_faces_dir(cropped_faces_dir)
        self._cropped_faces_dir = cropped_faces_dir

    def get_cropped_face_url(self, face_uuid: str) -> str:
        faces_filenames = os.listdir(self._cropped_faces_dir)

        face_url = None
        for filename in faces_filenames:
            if filename.startswith(face_uuid):
                face_url = filename

        if face_url is None:
            raise FileNotFoundError(f'Face not found.UUID={face_uuid}')

        face_url = os.path.join(self._cropped_faces_dir, face_url)

        return face_url

    @staticmethod
    def _preprocess_cropped_faces_dir(cropped_faces_dir):
        os.makedirs(cropped_faces_dir, exist_ok=True)
