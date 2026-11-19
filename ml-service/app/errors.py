"""Domain exceptions mapped to HTTP responses in main.py."""


class ServiceError(Exception):
    status_code = 400

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ValidationFailed(ServiceError):
    status_code = 422


class NotFound(ServiceError):
    status_code = 404


class PayloadTooLarge(ServiceError):
    status_code = 413


class ServiceUnavailable(ServiceError):
    status_code = 503
