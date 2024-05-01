import pytest


def test_package_installed():
    try:
        import test_package
    except ImportError:
        pytest.fail("Failed to import mypackage")
