import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TFHUB_CACHE_DIR"] = "/home/isr/app/.cache/tfhub_modules"

def test_load_metrabs():
    from pose_pipeline.wrappers.bridging import get_model
    model = get_model()

    assert model is not None, "Model not loaded correctly"