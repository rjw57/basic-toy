import pytest

from rwbasic.interpreter import Interpreter


@pytest.fixture
def interpreter():
    """Interpreter instance."""
    return Interpreter()
