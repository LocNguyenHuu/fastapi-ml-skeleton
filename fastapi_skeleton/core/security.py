import secrets
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED

from fastapi_skeleton.core import config
from fastapi_skeleton.core.messages import AUTH_REQ, NO_API_KEY

api_key = APIKeyHeader(name="token", auto_error=False)


def validate_request(header: Optional[str] = Security(api_key)) -> bool:
    """
    Validates the request by checking the provided header against the API key.

    Args:
        header (Optional[str]): The header containing the API key. Defaults to Security(api_key).

    Returns:
        bool: True if the request is valid, False otherwise.

    Raises:
        HTTPException: If the header is missing or the API key is incorrect.
    """
    if header is None:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, detail=NO_API_KEY, headers={}
        )
    if not secrets.compare_digest(header, str(config.API_KEY)):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail=AUTH_REQ, headers={}
        )
    return True
