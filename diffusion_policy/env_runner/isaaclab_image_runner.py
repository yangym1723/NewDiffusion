"""
IsaacLab Image Runner for Diffusion Policy evaluation.

This runner creates an IsaacLab ThreeFingers environment and runs rollout
evaluation during training and in eval.py. It extracts observations
(camera_rgb, camera_depth, ee_pose, ee_quat, contact_force_z) from the
IsaacLab env and feeds them to the diffusion policy in the expected format.

IMPORTANT: IsaacLab requires that the AppLauncher is initialized before
any other Isaac Sim imports. This runner handles that by lazily launching
the simulation on the first call to run().
"""

import os
import collections
import pathlib
import numpy as np
import torch
import tqdm
import cv2

from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

try:
    import wandb
    import wandb.sdk.data_types.video as wv
except ImportError:
    wandb = None
    wv = None


class IsaacLabImageRunner(BaseImageRunner):
    """
    Env runner that uses an IsaacLab (Isaac Sim) environment for evaluation.

    This runner:
    1. Creates a ThreeFingers IsaacLab env (ManagerBasedRLEnv)
    2. Runs rollout episodes using the diffusion policy
    3. Records videos and computes reward metrics
    4. Returns a log dict compatible with wandb logging

    Because IsaacLab environments are vectorized GPU environments (not
    standard gym envs), this runner does NOT use AsyncVectorEnv or
    MultiStepWrapper. Instead, it directly manages the observation
    stacking and action chunking logic.
    """

    def __init__(
        self,
        output_dir: str,
        shape_meta: dict,
        task_name: str = "Template-Threefingers-v0",
        n_test: int = 10,
        n_test_vis: int = 3,
        test_start_seed: int = 10000,
        max_steps: int = 750,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        past_action: bool = False,
        tqdm_interval_sec: float = 5.0,
        n_envs: int = 1,
        image_shape: tuple = (3, 240, 320),
        depth_shape: tuple = (1, 240, 320),
        enable_cameras: bool = True,
        headless: bool = True,
        device: str = "cuda:0",
    ):
        super().__init__(output_dir)

        self.shape_meta = shape_meta
        self.task_name = task_name
        self.n_test = n_test
        self.n_test_vis = n_test_vis
        self.test_start_seed = test_start_seed
        self.max_steps = max_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.fps = fps
        self.crf = crf
        self.past_action = past_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.n_envs = n_envs
        self.image_shape = tuple(image_shape)
        self.depth_shape = tuple(depth_shape)
        self.enable_cameras = enable_cameras
        self.headless = headless
        self.sim_device = device

        # Lazy initialization - IsaacLab env is created on first run()
        self._env = None
        self._simulation_app = None
        self._isaaclab_initialized = False

    def _ensure_isaaclab_initialized(self):
        """Lazily initialize IsaacLab and create the environment.

        IsaacLab requires AppLauncher to be created before any other
        Isaac Sim imports. We handle this by deferring initialization
        to the first run() call.
        """
        if self._isaaclab_initialized:
            return

        # Step 1: Launch the simulation app (MUST happen before other imports)
        from isaaclab.app import AppLauncher

        launcher_args = {
            "headless": self.headless,
            "enable_cameras": self.enable_cameras,
        }
        app_launcher = AppLauncher(launcher_args=launcher_args)
        self._simulation_app = app_launcher.app

        # Step 2: Now import IsaacLab modules (after AppLauncher is created)
        import gymnasium as gym

        # Import ThreeFingers package to trigger gym.register()
        # This import must happen AFTER AppLauncher initialization
        try:
            import ThreeFingers  # noqa: F401 - triggers gym.register()
        except ImportError:
            # If ThreeFingers is not installed as a package, try adding to path
            import sys
            threefingers_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))))),
                "ThreeFingers", "source", "ThreeFingers"
            )
            if os.path.exists(threefingers_path):
                sys.path.insert(0, threefingers_path)
                import ThreeFingers  # noqa: F401

        # Step 3: Create the environment
        from isaaclab_tasks.utils import parse_env_cfg

        env_cfg = parse_env_cfg(
            self.task_name,
            device=self.sim_device,
            num_envs=self.n_envs,
        )
        # Override episode length to match our max_steps
        # IsaacLab env has decimation=2, dt=1/120, so control rate is 60Hz
        # max_steps in diffusion policy context = number of action chunks
        # Each action chunk executes n_action_steps actions
        # We set a generous episode length
        env_cfg.episode_length_s = (self.max_steps * self.n_action_steps) / 60.0 + 5.0

        self._env = gym.make(self.task_name, cfg=env_cfg)
        self._isaaclab_initialized = True

    def _extract_obs(self, isaac_obs: dict, env) -> dict:
        """Extract and format observations from IsaacLab env output
        into the format expected by diffusion policy.

        IsaacLab returns:
            obs = {
                "policy": tensor(num_envs, D),  # concatenated low-dim
                "camera": {"rgb": tensor(N,H,W,3), "depth": tensor(N,H,W,1)}
            }

        Diffusion policy expects per-key observations:
            {
                "camera_rgb": np.array(N, C, H', W'),     # CHW, float32 [0,1]
                "camera_depth": np.array(N, 1, H', W'),   # CHW, float32
                "ee_pose": np.array(N, 3),
                "ee_quat": np.array(N, 4),
                "contact_force_z": np.array(N, 3),
            }
        """
        unwrapped_env = env.unwrapped
        result = {}

        # --- Extract low-dim observations directly from the env sensors ---
        # ee_pose: end-effector position in robot root frame (3,)
        from isaaclab.utils.math import subtract_frame_transforms
        ee_frame = unwrapped_env.scene["ee_frame"]
        robot = unwrapped_env.scene["robot"]
        ee_pos, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            ee_frame.data.target_pos_w[:, 0, :]
        )
        result["ee_pose"] = ee_pos.cpu().numpy().astype(np.float32)

        # ee_quat: end-effector quaternion in world frame (4,)
        ee_quat = ee_frame.data.target_quat_w[:, 0, :]
        result["ee_quat"] = ee_quat.cpu().numpy().astype(np.float32)

        # contact_force_z: Z-axis contact forces from 3 finger sensors (3,)
        force_z_list = []
        for name in ["contact_sensor_link1", "contact_sensor_link2", "contact_sensor_link3"]:
            sensor = unwrapped_env.scene.sensors.get(name)
            if sensor is not None:
                force_z = sensor.data.net_forces_w[:, :, 2].sum(dim=-1)
                force_z_list.append(force_z)
            else:
                force_z_list.append(torch.zeros(unwrapped_env.num_envs,
                                                device=unwrapped_env.device))
        contact_force = torch.stack(force_z_list, dim=-1).to(dtype=torch.float32)
        result["contact_force_z"] = contact_force.cpu().numpy()

        # --- Extract camera observations ---
        if "camera" in isaac_obs:
            camera_obs = isaac_obs["camera"]
            # RGB: IsaacLab returns (N, H, W, 3) uint8 -> convert to (N, C, H', W') float32 [0,1]
            if "rgb" in camera_obs:
                rgb = camera_obs["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                # Only take RGB channels (ignore alpha if present)
                if rgb.shape[-1] == 4:
                    rgb = rgb[..., :3]
                # HWC -> CHW
                rgb = np.transpose(rgb, (0, 3, 1, 2)).astype(np.float32) / 255.0
                # Resize to target shape if needed
                target_h, target_w = self.image_shape[1], self.image_shape[2]
                if rgb.shape[2] != target_h or rgb.shape[3] != target_w:
                    resized = np.zeros(
                        (rgb.shape[0], 3, target_h, target_w), dtype=np.float32
                    )
                    for i in range(rgb.shape[0]):
                        img = np.transpose(rgb[i], (1, 2, 0))  # CHW -> HWC
                        img = cv2.resize(img, (target_w, target_h))
                        resized[i] = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    rgb = resized
                result["camera_rgb"] = rgb

            # Depth: IsaacLab returns (N, H, W, 1) float -> convert to (N, 1, H', W')
            if "depth" in camera_obs:
                depth = camera_obs["depth"]
                if isinstance(depth, torch.Tensor):
                    depth = depth.cpu().numpy()
                # HWC -> CHW
                depth = np.transpose(depth, (0, 3, 1, 2)).astype(np.float32)
                # Resize to target shape if needed
                target_h, target_w = self.depth_shape[1], self.depth_shape[2]
                if depth.shape[2] != target_h or depth.shape[3] != target_w:
                    resized = np.zeros(
                        (depth.shape[0], 1, target_h, target_w), dtype=np.float32
                    )
                    for i in range(depth.shape[0]):
                        d = depth[i, 0]  # (H, W)
                        d = cv2.resize(d, (target_w, target_h))
                        resized[i, 0] = d
                    depth = resized
                result["camera_depth"] = depth
        else:
            # Fallback: camera obs not in the returned dict
            # This can happen if the observation groups don't include "camera"
            # Try to read directly from camera sensor
            camera_sensor = unwrapped_env.scene.sensors.get("camera")
            if camera_sensor is not None:
                rgb_data = camera_sensor.data.output.get("rgb")
                if rgb_data is not None:
                    rgb = rgb_data.cpu().numpy()
                    if rgb.shape[-1] == 4:
                        rgb = rgb[..., :3]
                    rgb = np.transpose(rgb, (0, 3, 1, 2)).astype(np.float32) / 255.0
                    target_h, target_w = self.image_shape[1], self.image_shape[2]
                    if rgb.shape[2] != target_h or rgb.shape[3] != target_w:
                        resized = np.zeros(
                            (rgb.shape[0], 3, target_h, target_w), dtype=np.float32
                        )
                        for i in range(rgb.shape[0]):
                            img = np.transpose(rgb[i], (1, 2, 0))
                            img = cv2.resize(img, (target_w, target_h))
                            resized[i] = np.transpose(img, (2, 0, 1))
                        rgb = resized
                    result["camera_rgb"] = rgb

                depth_data = camera_sensor.data.output.get("distance_to_camera")
                if depth_data is not None:
                    depth = depth_data.cpu().numpy()
                    if depth.ndim == 3:
                        depth = depth[..., np.newaxis]
                    depth = np.transpose(depth, (0, 3, 1, 2)).astype(np.float32)
                    target_h, target_w = self.depth_shape[1], self.depth_shape[2]
                    if depth.shape[2] != target_h or depth.shape[3] != target_w:
                        resized = np.zeros(
                            (depth.shape[0], 1, target_h, target_w), dtype=np.float32
                        )
                        for i in range(depth.shape[0]):
                            d = depth[i, 0]
                            d = cv2.resize(d, (target_w, target_h))
                            resized[i, 0] = d
                        depth = resized
                    result["camera_depth"] = depth

        return result

    def _obs_to_stacked(self, obs_history: list, n_steps: int) -> dict:
        """Stack the last n_steps observations into the format expected by
        the diffusion policy: {key: np.array(B, T, *shape)}.

        If fewer than n_steps observations are available, pad by repeating
        the earliest observation (same behavior as MultiStepWrapper).
        """
        result = {}
        keys = obs_history[-1].keys()
        for key in keys:
            all_obs = [obs[key] for obs in obs_history]
            n_available = len(all_obs)
            latest_shape = all_obs[-1].shape  # (B, *obs_shape)

            # Create output array: (B, n_steps, *obs_shape)
            batch_size = latest_shape[0]
            obs_shape = latest_shape[1:]
            stacked = np.zeros(
                (batch_size, n_steps) + obs_shape, dtype=all_obs[-1].dtype
            )

            # Fill from the end
            start_idx = max(0, n_steps - n_available)
            src_start = max(0, n_available - n_steps)
            for t_idx, src_idx in enumerate(range(src_start, n_available)):
                stacked[:, start_idx + t_idx] = all_obs[src_idx]

            # Pad the beginning by repeating the earliest available obs
            if start_idx > 0:
                for t_idx in range(start_idx):
                    stacked[:, t_idx] = stacked[:, start_idx]

            result[key] = stacked
        return result

    def _save_video(self, frames: list, file_path: str):
        """Save a list of RGB frames (HWC, uint8) to an MP4 video file."""
        if len(frames) == 0:
            return

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(file_path, fourcc, self.fps, (w, h))
        for frame in frames:
            # RGB -> BGR for OpenCV
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()

    def run(self, policy: BaseImagePolicy) -> Dict:
        """Run evaluation rollouts in IsaacLab environment.

        For each test episode:
        1. Reset the environment
        2. Run the policy in a loop, feeding stacked observations
        3. Record video frames (for the first n_test_vis episodes)
        4. Collect rewards

        Returns a dict of loggable data for wandb.
        """
        # Ensure IsaacLab is initialized
        self._ensure_isaaclab_initialized()

        device = policy.device
        env = self._env

        log_data = {}
        all_rewards = []
        max_rewards = collections.defaultdict(list)

        for ep_idx in range(self.n_test):
            seed = self.test_start_seed + ep_idx
            enable_render = ep_idx < self.n_test_vis

            # Reset env
            isaac_obs, info = env.reset()
            obs = self._extract_obs(isaac_obs, env)
            obs_history = [obs]

            # Track episode
            episode_rewards = []
            video_frames = []
            done = False
            step_count = 0

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval ThreeFingers ep {ep_idx+1}/{self.n_test}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            while not done and step_count < self.max_steps:
                # Stack observations for policy input
                stacked_obs = self._obs_to_stacked(obs_history, self.n_obs_steps)

                # Convert to torch tensors on policy device
                obs_dict = {}
                for key, val in stacked_obs.items():
                    obs_dict[key] = torch.from_numpy(val).to(device=device, dtype=torch.float32)

                # Run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # Get action chunk: (B, Ta, Da)
                action = action_dict["action"].detach().cpu().numpy()

                # Execute each action in the chunk
                for act_idx in range(min(self.n_action_steps, action.shape[1])):
                    if done:
                        break

                    single_action = action[:, act_idx, :]  # (B, Da)
                    single_action_tensor = torch.from_numpy(single_action).to(
                        device=env.unwrapped.device, dtype=torch.float32
                    )

                    isaac_obs, reward, terminated, truncated, info = env.step(single_action_tensor)

                    obs = self._extract_obs(isaac_obs, env)
                    # Keep obs_history bounded
                    obs_history.append(obs)
                    if len(obs_history) > self.n_obs_steps + 1:
                        obs_history.pop(0)

                    # Collect reward
                    if isinstance(reward, torch.Tensor):
                        reward_val = reward.cpu().numpy()
                    else:
                        reward_val = np.array(reward)
                    episode_rewards.append(reward_val.mean())

                    # Check done
                    if isinstance(terminated, torch.Tensor):
                        terminated = terminated.cpu().numpy()
                    if isinstance(truncated, torch.Tensor):
                        truncated = truncated.cpu().numpy()
                    done = np.any(terminated) or np.any(truncated)

                    # Record video frame
                    if enable_render:
                        try:
                            frame = env.render()
                            if isinstance(frame, torch.Tensor):
                                frame = frame.cpu().numpy()
                            if frame is not None:
                                if frame.dtype != np.uint8:
                                    frame = (frame * 255).astype(np.uint8)
                                # Handle batch dimension
                                if frame.ndim == 4:
                                    frame = frame[0]
                                video_frames.append(frame)
                        except Exception:
                            # Rendering may fail in some configurations
                            pass

                step_count += 1
                pbar.update(1)

            pbar.close()

            # Compute episode max reward
            if len(episode_rewards) > 0:
                max_reward = float(np.max(episode_rewards))
            else:
                max_reward = 0.0

            all_rewards.append(max_reward)
            max_rewards["test/"].append(max_reward)
            log_data[f"test/sim_max_reward_{seed}"] = max_reward

            # Save video
            if enable_render and len(video_frames) > 0 and wandb is not None:
                video_dir = pathlib.Path(self.output_dir) / "media"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_path = str(video_dir / f"{wv.util.generate_id()}.mp4")
                self._save_video(video_frames, video_path)
                log_data[f"test/sim_video_{seed}"] = wandb.Video(video_path)

        # Aggregate metrics
        for prefix, rewards in max_rewards.items():
            log_data[prefix + "mean_score"] = float(np.mean(rewards))

        return log_data

    def __del__(self):
        """Clean up IsaacLab resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
