import pytest

from rwbasic.interpreter import Interpreter


@pytest.fixture(scope="session")
def interpreter():
    """Shared interpreter for entire test suite. Good for tests which do not change state."""
    return Interpreter()
