# This script confirms that the mmlab packages are installed correctly.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_mmengine():
    import mmengine

    assert hasattr(mmengine, '__version__'), "mmengine is not installed correctly"
    assert isinstance(mmengine.__version__, str), "mmengine version is not a string"
    assert len(mmengine.__version__) > 0, "mmengine version string is empty"