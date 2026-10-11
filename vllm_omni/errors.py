# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-scoped client error types shared across vLLM-Omni entrypoints."""

from __future__ import annotations

from collections.abc import Callable
from http import HTTPStatus
from typing import NoReturn

DEFAULT_CLIENT_ERROR_TYPE = "BadRequestError"


class OmniClientError(ValueError):
    """
    Request-scoped error with an explicit HTTP status.
    One example of using OmniClientError is GuardrailViolationError, which is captured and resurfaced
    as HTTP 400 error code, instead of a generic 500. A service-unavailable
    pipeline route uses 503 without marking the whole engine dead.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = HTTPStatus.BAD_REQUEST.value,
        error_type: str = DEFAULT_CLIENT_ERROR_TYPE,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = int(status_code)
        self.error_type = error_type


class GuardrailViolationError(OmniClientError):
    """Raised when a model guardrail rejects request content."""


def client_error_metadata(exc: BaseException) -> tuple[int | None, str | None]:
    if isinstance(exc, OmniClientError):
        return exc.status_code, exc.error_type
    return None, None


def client_error_from_metadata(
    message: str,
    *,
    status_code: int | None,
    error_type: str | None,
) -> OmniClientError:
    return OmniClientError(
        message,
        status_code=status_code or HTTPStatus.BAD_REQUEST.value,
        error_type=error_type or DEFAULT_CLIENT_ERROR_TYPE,
    )


def is_client_error_status(status_code: int | None) -> bool:
    return status_code is not None and 400 <= int(status_code) < 500


def raise_client_error_or(
    message: str,
    *,
    status_code: int | None,
    error_type: str | None,
    fallback: Callable[[str], BaseException],
) -> NoReturn:
    """Raise a client error for 4xx statuses, otherwise raise ``fallback(message)``.

    Centralizes the "client-error-or-fallback" decision shared by the engine
    error paths so the status mapping lives in one place.
    """
    if is_client_error_status(status_code) or (
        status_code == HTTPStatus.SERVICE_UNAVAILABLE.value and error_type == "PipelineUnavailableError"
    ):
        raise client_error_from_metadata(
            message,
            status_code=status_code,
            error_type=error_type,
        )
    raise fallback(message)
