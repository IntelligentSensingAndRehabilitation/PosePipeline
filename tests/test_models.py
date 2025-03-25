import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def test_load_metrabs():
    from pose_pipeline.wrappers.bridging import get_model
    model = get_model()

    assert model is not None, "Model not loaded correctly"

def test_load_mmpose():
    from mmpose.apis import init_model as init_pose_estimator
    from mmpose.apis import inference_topdown