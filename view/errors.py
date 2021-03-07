UNKNOWN_ERROR = 0x1
MORE_THAN_ONE_FACE = 0x2

ERROR_CODE_TO_MESSAGE = {
    UNKNOWN_ERROR: 'Unknown error',
    MORE_THAN_ONE_FACE: 'There\'s more than one face on image'
}


class Error:
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

    @staticmethod
    def from_error_code(code: int):
        if code not in ERROR_CODE_TO_MESSAGE.keys():
            code = UNKNOWN_ERROR
        message = ERROR_CODE_TO_MESSAGE[code]

        return Error(code, message)

    @staticmethod
    def more_than_one_face():
        return Error.from_error_code(MORE_THAN_ONE_FACE)

    @staticmethod
    def unknown_error():
        return Error.from_error_code(UNKNOWN_ERROR)
