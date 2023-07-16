import typing

from numpy import int32

BasicValue = typing.Union[int32, float, str]


def is_integer_basic_value(value: BasicValue) -> typing.TypeGuard[int32]:
    """
    Return True if and only if value is an integer BASIC value.
    """
    return isinstance(value, int32)


def is_numeric_basic_value(value: BasicValue) -> typing.TypeGuard[typing.Union[int32, float]]:
    """
    Return True if and only if value is a numeric BASIC value.
    """
    return is_integer_basic_value(value) or isinstance(value, float)


def is_string_basic_value(value: BasicValue) -> typing.TypeGuard[str]:
    """
    Return True if and only if value is a string BASIC value.
    """
    return isinstance(value, str)


def is_basic_value(value: BasicValue) -> typing.TypeGuard[BasicValue]:
    """
    Return True if and only if value is a BASIC value.
    """
    return is_numeric_basic_value(value) or is_string_basic_value(value)


def basic_bool(value: bool) -> BasicValue:
    """
    Convert a Python boolean into a BASIC boolean.
    """
    return int32(-1) if value else int32(0)
