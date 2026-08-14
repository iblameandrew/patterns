import attention_algebra as aa


def test_public_exports():
    for name in aa.__all__:
        assert hasattr(aa, name), name


def test_version():
    assert aa.__version__ == "0.7.0"
