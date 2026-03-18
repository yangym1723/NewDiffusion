import h5py
import numpy as np

from diffusion_policy.dataset.robomimic_replay_image_dataset import (
    RobomimicReplayImageDataset,
)


def test_single_frame_image_obs_repeated_from_hdf5(tmp_path):
    hdf5_path = tmp_path / "single_frame_obs.hdf5"

    rgb_frame = np.array(
        [
            [[0, 10, 20], [30, 40, 50]],
            [[60, 70, 80], [90, 100, 110]],
        ],
        dtype=np.uint8,
    )
    depth_frame = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float32,
    )

    with h5py.File(hdf5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")
        demo.create_dataset("actions", data=np.arange(8, dtype=np.float32).reshape(4, 2))

        obs = demo.create_group("obs")
        obs.create_dataset("ee_pose", data=np.arange(12, dtype=np.float32).reshape(4, 3))
        obs.create_dataset("camera_rgb", data=rgb_frame[None, ...])
        obs.create_dataset("camera_depth", data=depth_frame[None, ...])

    shape_meta = {
        "obs": {
            "camera_rgb": {
                "shape": [3, 2, 2],
                "type": "rgb",
            },
            "camera_depth": {
                "shape": [1, 2, 2],
                "type": "depth",
            },
            "ee_pose": {
                "shape": [3],
                "type": "low_dim",
            },
        },
        "actions": {
            "shape": [2],
            "binary_dims": [],
        },
    }

    dataset = RobomimicReplayImageDataset(
        shape_meta=shape_meta,
        dataset_path=str(hdf5_path),
        horizon=4,
        pad_before=1,
        pad_after=0,
        n_obs_steps=2,
        use_cache=False,
        load_image_obs_from_hdf5=True,
        repeat_single_frame_image_obs=True,
        val_ratio=0.0,
    )

    assert "camera_rgb" not in dataset.replay_buffer
    assert "camera_depth" not in dataset.replay_buffer

    normalizer_stats = dataset.get_normalizer().get_input_stats()
    assert "camera_depth" in normalizer_stats

    sample = dataset[0]
    obs = sample["obs"]
    rgb = obs["camera_rgb"].numpy()
    depth = obs["camera_depth"].numpy()

    expected_rgb = np.moveaxis(rgb_frame, -1, 0).astype(np.float32) / 255.0
    expected_depth = depth_frame[None, ...].astype(np.float32)

    assert rgb.shape == (2, 3, 2, 2)
    assert depth.shape == (2, 1, 2, 2)
    np.testing.assert_allclose(rgb[0], expected_rgb)
    np.testing.assert_allclose(rgb[1], expected_rgb)
    np.testing.assert_allclose(depth[0], expected_depth)
    np.testing.assert_allclose(depth[1], expected_depth)
