"""Unit tests for core API response schemas and error codes."""

import pytest

from baize.core.schemas import ApiResponse, ErrorCode, error, success


def test_error_code_values():
    assert ErrorCode.SUCCESS == 0
    assert ErrorCode.BAD_REQUEST == 40000
    assert ErrorCode.UNAUTHORIZED == 40100
    assert ErrorCode.FORBIDDEN == 40300
    assert ErrorCode.NOT_FOUND == 40400
    assert ErrorCode.INTERNAL_ERROR == 50000


def test_success_factory_with_data():
    result = success({"key": "value"})
    assert result.code == 0
    assert result.data == {"key": "value"}
    assert result.message == "ok"


def test_success_factory_with_none():
    result = success(None)
    assert result.code == 0
    assert result.data is None


def test_success_factory_with_list():
    result = success([1, 2, 3])
    assert result.code == 0
    assert result.data == [1, 2, 3]


@pytest.mark.parametrize(
    "code,expected_code",
    [
        (ErrorCode.BAD_REQUEST, 40000),
        (ErrorCode.UNAUTHORIZED, 40100),
        (ErrorCode.FORBIDDEN, 40300),
        (ErrorCode.NOT_FOUND, 40400),
        (ErrorCode.INTERNAL_ERROR, 50000),
    ],
)
def test_error_factory(code, expected_code):
    result = error(code, "something went wrong")
    assert result.code == expected_code
    assert result.data is None
    assert result.message == "something went wrong"


def test_api_response_generic_type():
    """ApiResponse data field accepts any type."""
    str_result: ApiResponse[str] = success("hello")
    assert str_result.data == "hello"
    assert str_result.code == 0


def test_error_factory_data_is_none():
    result = error(ErrorCode.NOT_FOUND, "not found")
    assert result.data is None
