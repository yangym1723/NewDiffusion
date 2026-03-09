"""
Evaluation script for diffusion policy with IsaacLab simulation.

Unlike the standard eval.py, this script initializes IsaacLab's AppLauncher
BEFORE any other imports, which is required by Isaac Sim's architecture.

Usage:
    python eval_isaaclab.py -c <checkpoint_path> -o <output_dir> \
        [--task Template-Threefingers-v0] [--num_envs 1] [--headless]

Example:
    python eval_isaaclab.py \
        -c data/outputs/2024.01.01/12.00.00_train_diffusion_transformer_hybrid_real_image/checkpoints/latest.ckpt \
        -o data/eval_output \
        --headless
# 使用专用的 IsaacLab eval 脚本
python eval_isaaclab.py \
    -c path/to/your/checkpoint.ckpt \
    -o data/eval_output \
    --task Template-Threefingers-v0 \
    --headless \
    --n_test 10 \
    --n_test_vis 3
"""

import sys
# Use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import argparse

# ============================================================
# STEP 1: Initialize IsaacLab AppLauncher BEFORE anything else
# This MUST happen before any other Isaac Sim / IsaacLab imports
# ============================================================
parser = argparse.ArgumentParser(description="Evaluate diffusion policy in IsaacLab simulation")
parser.add_argument("-c", "--checkpoint", required=True, help="Path to .ckpt checkpoint file")
parser.add_argument("-o", "--output_dir", required=True, help="Output directory for eval results")
parser.add_argument("-d", "--policy_device", default="cuda:0", help="Device for policy inference")
parser.add_argument("--task", default="Template-Threefingers-v0", help="IsaacLab task gym ID")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--n_test", type=int, default=10, help="Number of test episodes")
parser.add_argument("--n_test_vis", type=int, default=3, help="Number of test episodes to record video")
parser.add_argument("--max_steps", type=int, default=750, help="Max action chunks per episode")

# Add IsaacLab AppLauncher arguments
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# Parse all args
args = parser.parse_args()
# Ensure cameras are enabled (needed for camera observations)
args.enable_cameras = True

# Launch the simulation app
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ============================================================
# STEP 2: Now safe to import everything else
# ============================================================
import os
import pathlib
import json
import collections

import hydra
import torch
import dill
import numpy as np
import cv2
import tqdm

try:
    import wandb
    import wandb.sdk.data_types.video as wv
except ImportError:
    wandb = None
    wv = None

import gymnasium as gym

# Import ThreeFingers to trigger gym.register()
try:
    import ThreeFingers  # noqa: F401
except ImportError:
    threefingers_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "ThreeFingers", "source", "ThreeFingers"
    )
    threefingers_path = os.path.normpath(threefingers_path)
    if os.path.exists(threefingers_path) and threefingers_path not in sys.path:
        sys.path.insert(0, threefingers_path)
        import ThreeFingers  # noqa: F401

from isaaclab.utils.math import subtract_frame_transforms
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def extract_obs(isaac_obs, env, image_shape, depth_shape):
    """Extract observations from IsaacLab env into diffusion policy format.

    Returns a dict with keys matching the shape_meta:
        camera_rgb:      np.array (N, 3, H, W) float32 in [0,1]
        camera_depth:    np.array (N, 1, H, W) float32
        ee_pos:          np.array (N, 3) float32
        ee_quat:         np.array (N, 4) float32
        contact_force_z: np.array (N, 3) float32
    """
    unwrapped = env.unwrapped
    result = {}

    # -- Low-dim observations (read directly from scene sensors) --
    ee_frame = unwrapped.scene["ee_frame"]
    robot = unwrapped.scene["robot"]

    # ee_pos: EE position in robot root frame
    ee_pos, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        ee_frame.data.target_pos_w[:, 0, :]
    )
    result["ee_pos"] = ee_pos.cpu().numpy().astype(np.float32)

    # ee_quat: EE quaternion in world frame
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    result["ee_quat"] = ee_quat.cpu().numpy().astype(np.float32)

    # contact_force_z: Z-force per finger sensor
    force_z_list = []
    for name in ["contact_sensor_link1", "contact_sensor_link2", "contact_sensor_link3"]:
        sensor = unwrapped.scene.sensors.get(name)
        if sensor is not None:
            fz = sensor.data.net_forces_w[:, :, 2].sum(dim=-1)
            force_z_list.append(fz)
        else:
            force_z_list.append(torch.zeros(unwrapped.num_envs, device=unwrapped.device))
    contact_force = torch.stack(force_z_list, dim=-1).float()
    result["contact_force_z"] = contact_force.cpu().numpy()

    # -- Camera observations --
    target_h_rgb, target_w_rgb = image_shape[1], image_shape[2]
    target_h_d, target_w_d = depth_shape[1], depth_shape[2]

    # Try to get camera data from the obs dict first, then fallback to sensor
    camera_obs = isaac_obs.get("camera", None)
    if camera_obs is not None and isinstance(camera_obs, dict):
        rgb_raw = camera_obs.get("rgb", None)
        depth_raw = camera_obs.get("depth", None)
    else:
        # Read directly from sensor
        cam_sensor = unwrapped.scene.sensors.get("camera")
        rgb_raw = cam_sensor.data.output.get("rgb") if cam_sensor else None
        depth_raw = cam_sensor.data.output.get("distance_to_camera") if cam_sensor else None

    # Process RGB
    if rgb_raw is not None:
        if isinstance(rgb_raw, torch.Tensor):
            rgb_raw = rgb_raw.cpu().numpy()
        if rgb_raw.shape[-1] == 4:
            rgb_raw = rgb_raw[..., :3]
        # (N, H, W, 3) -> (N, 3, H, W) float32 [0, 1]
        rgb = np.transpose(rgb_raw, (0, 3, 1, 2)).astype(np.float32)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        # Resize if needed
        if rgb.shape[2] != target_h_rgb or rgb.shape[3] != target_w_rgb:
            resized = np.zeros((rgb.shape[0], 3, target_h_rgb, target_w_rgb), dtype=np.float32)
            for i in range(rgb.shape[0]):
                img_hwc = np.transpose(rgb[i], (1, 2, 0))
                img_hwc = cv2.resize(img_hwc, (target_w_rgb, target_h_rgb))
                resized[i] = np.transpose(img_hwc, (2, 0, 1))
            rgb = resized
        result["camera_rgb"] = rgb

    # Process Depth
    if depth_raw is not None:
        if isinstance(depth_raw, torch.Tensor):
            depth_raw = depth_raw.cpu().numpy()
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[..., np.newaxis]
        # (N, H, W, 1) -> (N, 1, H, W)
        depth = np.transpose(depth_raw, (0, 3, 1, 2)).astype(np.float32)
        if depth.shape[2] != target_h_d or depth.shape[3] != target_w_d:
            resized = np.zeros((depth.shape[0], 1, target_h_d, target_w_d), dtype=np.float32)
            for i in range(depth.shape[0]):
                d = cv2.resize(depth[i, 0], (target_w_d, target_h_d))
                resized[i, 0] = d
            depth = resized
        result["camera_depth"] = depth

    return result


def stack_obs(obs_history, n_steps):
    """Stack last n_steps observations into (B, T, *shape) arrays."""
    result = {}
    keys = obs_history[-1].keys()
    for key in keys:
        all_obs = [obs[key] for obs in obs_history]
        n_available = len(all_obs)
        latest_shape = all_obs[-1].shape
        batch_size = latest_shape[0]
        obs_shape = latest_shape[1:]

        stacked = np.zeros((batch_size, n_steps) + obs_shape, dtype=all_obs[-1].dtype)

        start_idx = max(0, n_steps - n_available)
        src_start = max(0, n_available - n_steps)
        for t_idx, src_idx in enumerate(range(src_start, n_available)):
            stacked[:, start_idx + t_idx] = all_obs[src_idx]

        # Pad beginning by repeating earliest available
        if start_idx > 0:
            for t_idx in range(start_idx):
                stacked[:, t_idx] = stacked[:, start_idx]

        result[key] = stacked
    return result


def save_video(frames, file_path, fps):
    """Save a list of RGB (HWC, uint8) frames to MP4."""
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))
    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def main():
    checkpoint_path = args.checkpoint
    output_dir = args.output_dir
    policy_device = args.policy_device
    task_name = args.task
    num_envs = args.num_envs
    n_test = args.n_test
    n_test_vis = args.n_test_vis
    max_steps = args.max_steps

    if os.path.exists(output_dir):
        response = input(f"Output path {output_dir} already exists! Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Load checkpoint and reconstruct policy
    # ============================================================
    print(f"Loading checkpoint: {checkpoint_path}")
    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(policy_device)
    policy.to(device)
    policy.eval()

    # Extract shape info from config
    shape_meta = cfg.task.shape_meta
    image_shape = tuple(shape_meta['obs']['camera_rgb']['shape'])  # (3, 240, 320)
    depth_shape = tuple(shape_meta['obs']['camera_depth']['shape'])  # (1, 240, 320)
    n_obs_steps = cfg.n_obs_steps
    n_action_steps = cfg.n_action_steps
    fps = 10

    # ============================================================
    # Create IsaacLab environment
    # ============================================================
    from isaaclab_tasks.utils import parse_env_cfg

    print(f"Creating IsaacLab environment: {task_name} (num_envs={num_envs})")
    env_cfg = parse_env_cfg(
        task_name,
        device=policy_device,
        num_envs=num_envs,
    )
    # Set a generous episode length
    env_cfg.episode_length_s = (max_steps * n_action_steps) / 60.0 + 5.0

    env = gym.make(task_name, cfg=env_cfg)

    # ============================================================
    # Run evaluation rollouts
    # ============================================================
    print(f"Running {n_test} evaluation episodes (recording video for first {n_test_vis})...")

    log_data = {}
    max_rewards = collections.defaultdict(list)
    test_start_seed = 10000

    for ep_idx in range(n_test):
        seed = test_start_seed + ep_idx
        enable_render = ep_idx < n_test_vis

        # Reset
        isaac_obs, info = env.reset()
        obs = extract_obs(isaac_obs, env, image_shape, depth_shape)
        obs_history = [obs]

        episode_rewards = []
        video_frames = []
        done = False
        step_count = 0

        pbar = tqdm.tqdm(
            total=max_steps,
            desc=f"Eval ThreeFingers ep {ep_idx+1}/{n_test}",
            leave=False,
            mininterval=5.0,
        )

        while not done and step_count < max_steps:
            # Stack observations
            stacked_obs = stack_obs(obs_history, n_obs_steps)

            # To torch
            obs_dict = {}
            for key, val in stacked_obs.items():
                obs_dict[key] = torch.from_numpy(val).to(device=device, dtype=torch.float32)

            # Policy inference
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)

            action = action_dict["action"].detach().cpu().numpy()  # (B, Ta, Da)

            # Execute action chunk
            for act_idx in range(min(n_action_steps, action.shape[1])):
                if done:
                    break

                single_action = action[:, act_idx, :]  # (B, Da)
                action_tensor = torch.from_numpy(single_action).to(
                    device=env.unwrapped.device, dtype=torch.float32
                )

                isaac_obs, reward, terminated, truncated, info = env.step(action_tensor)

                obs = extract_obs(isaac_obs, env, image_shape, depth_shape)
                obs_history.append(obs)
                if len(obs_history) > n_obs_steps + 1:
                    obs_history.pop(0)

                # Reward
                if isinstance(reward, torch.Tensor):
                    rew = reward.cpu().numpy()
                else:
                    rew = np.array(reward)
                episode_rewards.append(rew.mean())

                # Done check
                if isinstance(terminated, torch.Tensor):
                    terminated = terminated.cpu().numpy()
                if isinstance(truncated, torch.Tensor):
                    truncated = truncated.cpu().numpy()
                done = bool(np.any(terminated) or np.any(truncated))

                # Video recording
                if enable_render:
                    try:
                        frame = env.render()
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if frame is not None:
                            if frame.dtype != np.uint8:
                                if frame.max() <= 1.0:
                                    frame = (frame * 255).astype(np.uint8)
                                else:
                                    frame = frame.astype(np.uint8)
                            if frame.ndim == 4:
                                frame = frame[0]
                            video_frames.append(frame)
                    except Exception:
                        pass

            step_count += 1
            pbar.update(1)

        pbar.close()

        # Episode max reward
        if len(episode_rewards) > 0:
            max_reward = float(np.max(episode_rewards))
        else:
            max_reward = 0.0

        max_rewards["test/"].append(max_reward)
        log_data[f"test/sim_max_reward_{seed}"] = max_reward
        print(f"  Episode {ep_idx+1}: max_reward={max_reward:.4f}, "
              f"steps={len(episode_rewards)}, video_frames={len(video_frames)}")

        # Save video
        if enable_render and len(video_frames) > 0:
            video_dir = pathlib.Path(output_dir) / "media"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_filename = f"eval_ep{ep_idx}_seed{seed}.mp4"
            video_path = str(video_dir / video_filename)
            save_video(video_frames, video_path, fps)
            log_data[f"test/sim_video_{seed}"] = video_path
            print(f"    Video saved: {video_path}")

    # Aggregate metrics
    for prefix, rewards in max_rewards.items():
        mean_score = float(np.mean(rewards))
        log_data[prefix + "mean_score"] = mean_score
        print(f"\n{prefix}mean_score = {mean_score:.4f}")

    # Save eval_log.json
    json_log = {}
    for key, value in log_data.items():
        json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    print(f"\nEval log saved to: {out_path}")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == '__main__':
    main()
