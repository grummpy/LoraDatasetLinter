import pytest
from tests.fixtures import relax


@pytest.fixture
def policy():
    return relax()
