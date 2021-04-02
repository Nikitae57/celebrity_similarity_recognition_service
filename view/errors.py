from enum import Enum


class ErrorType(Enum):
    UNKNOWN_ERROR = 0x1
    MORE_THAN_ONE_FACE = 0x2
    NO_FACES = 0x3


ERROR_TYPE_TO_MESSAGE = {
    ErrorType.UNKNOWN_ERROR: 'Unknown error',
    ErrorType.MORE_THAN_ONE_FACE: 'There\'s more than one face on image',
    ErrorType.NO_FACES: 'No faces found on image'
}


class Error:
    def __init__(self, error_type: ErrorType, message: str):
        self.type = error_type.value
        self.message = message

    @staticmethod
    def from_error_code(code: ErrorType):
        if code not in ERROR_TYPE_TO_MESSAGE.keys():
            code = ErrorType.UNKNOWN_ERROR
        message = ERROR_TYPE_TO_MESSAGE[code]

        return Error(code, message)

    @staticmethod
    def more_than_one_face():
        return Error.from_error_code(ErrorType.MORE_THAN_ONE_FACE)

    @staticmethod
    def no_faces():
        return Error.from_error_code(ErrorType.NO_FACES)

    @staticmethod
    def unknown_error():
        return Error.from_error_code(ErrorType.UNKNOWN_ERROR)
