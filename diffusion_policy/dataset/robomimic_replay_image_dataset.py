from typing import Dict, List
import torch
import numpy as np
import cv2
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            load_image_obs_from_hdf5=False,
            repeat_single_frame_image_obs=False,
            preload_single_frame_image_obs=True,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(
                f"Dataset file not found: {dataset_path}. "
                "Please set task.dataset_path to the correct HDF5 file."
            )

        replay_buffer = None
        if use_cache:
            # Include shape_meta hash in cache filename to auto-invalidate
            # when data format (obs keys, action dims, binary_dims, etc.) changes.
            meta_for_hash = {
                'obs': {k: v for k, v in shape_meta['obs'].items()},
                'actions': shape_meta['actions'],
                'abs_action': abs_action,
                'rotation_rep': rotation_rep,
                'load_image_obs_from_hdf5': load_image_obs_from_hdf5,
                'repeat_single_frame_image_obs': repeat_single_frame_image_obs,
                'preload_single_frame_image_obs': preload_single_frame_image_obs,
            }
            meta_hash = hashlib.md5(
                json.dumps(meta_for_hash, sort_keys=True, default=str).encode()
            ).hexdigest()[:8]
            cache_zarr_path = dataset_path + f'.{meta_hash}.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer,
                            store_image_obs=not load_image_obs_from_hdf5,
                            repeat_single_frame_image_obs=repeat_single_frame_image_obs)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        if os.path.isdir(cache_zarr_path):
                            shutil.rmtree(cache_zarr_path)
                        elif os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer,
                store_image_obs=not load_image_obs_from_hdf5,
                repeat_single_frame_image_obs=repeat_single_frame_image_obs)

        rgb_keys = list()
        lowdim_keys = list()
        depth_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
            elif type == 'depth':
                depth_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys + depth_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        # Pre-compute per-step episode-level cumulative action sums.
        # cumact_cumsum[t] = sum of actions from episode start up to (but not
        # including) step t.  Resets to zero at each episode boundary.
        # Binary action dims (e.g. gripper on/off) are excluded from cumact
        # since their cumulative sum has no physical meaning.
        binary_dims = shape_meta['actions'].get('binary_dims', None)
        persistent_dims = shape_meta['actions'].get('persistent_dims', None)
        non_cumact_dims = list()
        if binary_dims is not None:
            non_cumact_dims.extend(binary_dims)
        if persistent_dims is not None:
            non_cumact_dims.extend(persistent_dims)
        non_cumact_dims = sorted(set(non_cumact_dims))
        all_actions = replay_buffer['actions'][:]  # (N_total, Da)
        episode_ends_np = replay_buffer.episode_ends[:]
        cumact_cumsum = np.zeros_like(all_actions)  # (N_total, Da)
        prev_end = 0
        for ep_end in episode_ends_np:
            ep_actions = all_actions[prev_end:ep_end].copy()  # (ep_len, Da)
            # Zero out non-cumulative dims before cumulative sum.
            if len(non_cumact_dims) > 0:
                for dim in non_cumact_dims:
                    ep_actions[:, dim] = 0.0
            # cumsum gives [a0, a0+a1, ...]; right-shift so offset[0]=0
            cs = np.cumsum(ep_actions, axis=0)
            # right-shift: offset[t] = sum(action[0:t])
            cumact_cumsum[prev_end] = 0.0
            if ep_end - prev_end > 1:
                cumact_cumsum[prev_end + 1:ep_end] = cs[:-1]
            prev_end = ep_end

        self.cumact_cumsum = cumact_cumsum
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.depth_keys = depth_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_path = dataset_path
        self.load_image_obs_from_hdf5 = load_image_obs_from_hdf5
        self.repeat_single_frame_image_obs = repeat_single_frame_image_obs
        self.preload_single_frame_image_obs = preload_single_frame_image_obs
        self.episode_ends = replay_buffer.episode_ends[:].astype(np.int64)
        self.episode_starts = np.concatenate([
            np.zeros((1,), dtype=np.int64),
            self.episode_ends[:-1]
        ]) if len(self.episode_ends) > 0 else np.zeros((0,), dtype=np.int64)
        self.episode_lengths = self.episode_ends - self.episode_starts
        self._hdf5_file = None
        self._image_obs_stats = None
        self._single_frame_image_cache = {
            key: dict() for key in (self.rgb_keys + self.depth_keys)
        }
        if self.load_image_obs_from_hdf5 \
                and self.repeat_single_frame_image_obs \
                and self.preload_single_frame_image_obs:
            self._preload_single_frame_image_obs()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        val_set._hdf5_file = None
        return val_set

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_hdf5_file'] = None
        return state

    def __del__(self):
        hdf5_file = getattr(self, '_hdf5_file', None)
        if hdf5_file is not None:
            try:
                hdf5_file.close()
            except Exception:
                pass

    def _get_hdf5_file(self):
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.dataset_path, 'r')
        return self._hdf5_file

    def _get_obs_steps(self):
        if self.n_obs_steps is None:
            return self.sampler.sequence_length
        return min(self.n_obs_steps, self.sampler.sequence_length)

    def _preload_single_frame_image_obs(self):
        image_keys = self.rgb_keys + self.depth_keys
        if len(image_keys) == 0:
            return

        with h5py.File(self.dataset_path, 'r') as file:
            demos = file['data']
            for episode_idx in range(len(self.episode_ends)):
                demo = demos[f'demo_{episode_idx}']
                episode_length = int(demo['actions'].shape[0])
                for key in image_keys:
                    hdf5_arr = demo['obs'][key]
                    n_frames = int(hdf5_arr.shape[0])
                    if n_frames == 1:
                        obs_type = self.shape_meta['obs'][key].get('type', 'low_dim')
                        target_shape = tuple(self.shape_meta['obs'][key]['shape'])
                        frame = self._prepare_image_frame(
                            hdf5_arr[0], target_shape=target_shape, obs_type=obs_type)
                        self._single_frame_image_cache[key][episode_idx] = frame
                    elif n_frames != episode_length:
                        raise ValueError(
                            f"data/demo_{episode_idx}/obs/{key} has {n_frames} frames, "
                            f"but actions has {episode_length}. "
                            "Expected either 1 frame or one frame per action step."
                        )

    def _get_sequence_meta(self, idx: int):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = \
            self.sampler.indices[idx]
        episode_idx = int(np.searchsorted(
            self.episode_ends, buffer_start_idx, side='right'))
        episode_start = int(self.episode_starts[episode_idx])
        return {
            'episode_idx': episode_idx,
            'buffer_start_idx': int(buffer_start_idx),
            'buffer_end_idx': int(buffer_end_idx),
            'sample_start_idx': int(sample_start_idx),
            'sample_end_idx': int(sample_end_idx),
            'episode_start': episode_start,
            'episode_length': int(self.episode_lengths[episode_idx]),
            'n_data': int(buffer_end_idx - buffer_start_idx),
            'episode_local_start': int(buffer_start_idx - episode_start),
        }

    def _get_source_indices(self, sequence_meta: dict, target_steps: int):
        source_indices = np.empty((target_steps,), dtype=np.int64)
        first_idx = sequence_meta['episode_local_start']
        last_idx = first_idx + sequence_meta['n_data'] - 1
        sample_start_idx = sequence_meta['sample_start_idx']
        sample_end_idx = sequence_meta['sample_end_idx']
        for step_idx in range(target_steps):
            if step_idx < sample_start_idx:
                source_indices[step_idx] = first_idx
            elif step_idx >= sample_end_idx:
                source_indices[step_idx] = last_idx
            else:
                source_indices[step_idx] = first_idx + (step_idx - sample_start_idx)
        return source_indices

    def _prepare_image_frame(self, frame, target_shape, obs_type: str):
        c, h, w = target_shape
        interpolation = cv2.INTER_AREA
        if obs_type == 'depth':
            interpolation = cv2.INTER_NEAREST
            frame = frame.astype(np.float32, copy=False)

        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]

        src_h, src_w = frame.shape[:2]
        if src_h != h or src_w != w:
            if frame.shape[-1] == 1:
                frame = cv2.resize(frame[:, :, 0], (w, h), interpolation=interpolation)
                frame = frame[:, :, np.newaxis]
            else:
                frame = cv2.resize(frame, (w, h), interpolation=interpolation)

        if frame.ndim == 2:
            frame = frame[:, :, np.newaxis]
        return frame

    def _load_image_sequence_from_hdf5(self, idx: int, key: str, obs_type: str):
        sequence_meta = self._get_sequence_meta(idx)
        target_steps = self._get_obs_steps()
        source_indices = self._get_source_indices(sequence_meta, target_steps)
        target_shape = tuple(self.shape_meta['obs'][key]['shape'])

        demo = self._get_hdf5_file()['data'][f"demo_{sequence_meta['episode_idx']}"]
        hdf5_arr = demo['obs'][key]
        n_frames = int(hdf5_arr.shape[0])

        if n_frames == 1:
            if not self.repeat_single_frame_image_obs:
                raise ValueError(
                    f"data/demo_{sequence_meta['episode_idx']}/obs/{key} only has 1 frame. "
                    "Set repeat_single_frame_image_obs=True to reuse it during training."
                )
            cached_frame = self._single_frame_image_cache[key].get(
                sequence_meta['episode_idx'])
            if cached_frame is None:
                cached_frame = self._prepare_image_frame(
                    hdf5_arr[0], target_shape=target_shape, obs_type=obs_type)
                self._single_frame_image_cache[key][sequence_meta['episode_idx']] = cached_frame
            return np.repeat(cached_frame[np.newaxis, ...], target_steps, axis=0)
        elif np.max(source_indices) >= n_frames:
            raise ValueError(
                f"data/demo_{sequence_meta['episode_idx']}/obs/{key} has {n_frames} frames, "
                f"but training needs index {int(np.max(source_indices))}. "
                "Expected either 1 frame or one frame per action step."
            )

        frames = [
            self._prepare_image_frame(
                hdf5_arr[int(src_idx)], target_shape=target_shape, obs_type=obs_type)
            for src_idx in source_indices
        ]
        return np.stack(frames, axis=0)

    def _get_image_obs_stats_from_hdf5(self):
        if self._image_obs_stats is not None:
            return self._image_obs_stats

        stats = dict()
        if len(self.depth_keys) == 0:
            self._image_obs_stats = stats
            return stats

        with h5py.File(self.dataset_path, 'r') as file:
            demos = file['data']
            for key in self.depth_keys:
                target_shape = tuple(self.shape_meta['obs'][key]['shape'])
                min_arr = None
                max_arr = None
                sum_arr = None
                sq_sum_arr = None
                count = 0

                for episode_idx in range(len(self.episode_ends)):
                    demo = demos[f'demo_{episode_idx}']
                    hdf5_arr = demo['obs'][key]
                    episode_length = int(demo['actions'].shape[0])
                    n_frames = int(hdf5_arr.shape[0])

                    if n_frames == 1:
                        if not self.repeat_single_frame_image_obs:
                            raise ValueError(
                                f"data/demo_{episode_idx}/obs/{key} only has 1 frame. "
                                "Set repeat_single_frame_image_obs=True to reuse it during training."
                            )
                        frame = self._prepare_image_frame(
                            hdf5_arr[0], target_shape=target_shape, obs_type='depth')
                        weight = episode_length
                        frame64 = frame.astype(np.float64)
                        if min_arr is None:
                            min_arr = frame.copy()
                            max_arr = frame.copy()
                            sum_arr = frame64 * weight
                            sq_sum_arr = np.square(frame64) * weight
                        else:
                            min_arr = np.minimum(min_arr, frame)
                            max_arr = np.maximum(max_arr, frame)
                            sum_arr += frame64 * weight
                            sq_sum_arr += np.square(frame64) * weight
                        count += weight
                        continue

                    if n_frames != episode_length:
                        raise ValueError(
                            f"data/demo_{episode_idx}/obs/{key} has {n_frames} frames, "
                            f"but actions has {episode_length}. "
                            "Expected either 1 frame or one frame per action step."
                        )

                    for frame_idx in range(n_frames):
                        frame = self._prepare_image_frame(
                            hdf5_arr[frame_idx], target_shape=target_shape, obs_type='depth')
                        frame64 = frame.astype(np.float64)
                        if min_arr is None:
                            min_arr = frame.copy()
                            max_arr = frame.copy()
                            sum_arr = frame64
                            sq_sum_arr = np.square(frame64)
                        else:
                            min_arr = np.minimum(min_arr, frame)
                            max_arr = np.maximum(max_arr, frame)
                            sum_arr += frame64
                            sq_sum_arr += np.square(frame64)
                        count += 1

                mean_arr = (sum_arr / count).astype(np.float32)
                var_arr = np.maximum(sq_sum_arr / count - np.square(mean_arr.astype(np.float64)), 0.0)
                stats[key] = {
                    'min': min_arr.astype(np.float32),
                    'max': max_arr.astype(np.float32),
                    'mean': mean_arr,
                    'std': np.sqrt(var_arr).astype(np.float32)
                }

        self._image_obs_stats = stats
        return stats

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['actions'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # Use range normalizer to map all action dims to [-1, 1].
            # This is important for diffusion model training (consistent scale).
            # For binary dims (e.g. gripper), min=0/max=1 naturally maps to {-1, +1}.
            this_normalizer = get_range_normalizer_from_stat(stat)
        normalizer['actions'] = this_normalizer

        # cumact (episode-level cumulative action sums)
        # Uses its own normalizer fitted on actual cumulative sum values,
        # NOT the action normalizer, because cumulative sums have a completely
        # different value range than individual actions.
        cumact_stat = array_to_stats(self.cumact_cumsum)
        normalizer['cumact'] = get_range_normalizer_from_stat(cumact_stat)

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pose'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('force_z'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        # depth
        for key in self.depth_keys:
            if key in self.replay_buffer:
                stat = array_to_stats(self.replay_buffer[key])
            else:
                stat = self._get_image_obs_stats_from_hdf5()[key]
            normalizer[key] = get_range_normalizer_from_stat(stat)

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['actions'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        obs_steps = self._get_obs_steps()
        T_slice = slice(obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            if self.load_image_obs_from_hdf5:
                image_data = self._load_image_sequence_from_hdf5(
                    idx, key=key, obs_type='rgb')
            else:
                image_data = data[key][T_slice]
                del data[key]
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(image_data, -1, 1).astype(np.float32) / 255.
            # T,C,H,W
        for key in self.depth_keys:
            if self.load_image_obs_from_hdf5:
                depth_data = self._load_image_sequence_from_hdf5(
                    idx, key=key, obs_type='depth')
            else:
                depth_data = data[key][T_slice]
                del data[key]
            # move channel last to channel first
            # T,H,W,C -> T,C,H,W, already float32
            obs_dict[key] = np.moveaxis(depth_data, -1, 1).astype(np.float32)
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        # Episode-level cumulative action offset: the sum of all actions from
        # episode start up to (but not including) the first step of this window.
        # Shape: (Da,) — a single vector per sample.
        buffer_start_idx = self.sampler.indices[idx][0]
        cumact_offset = self.cumact_cumsum[buffer_start_idx].astype(np.float32)

        # Number of left-padded steps (replicated first action).
        # Used by compute_loss to zero out padded positions before cumact cumsum.
        pad_left = int(self.sampler.indices[idx][2])  # sample_start_idx

        # Number of right-padded steps (replicated last action).
        # Used by compute_loss to zero out padded positions before cumact cumsum.
        pad_right = int(self.sampler.sequence_length - self.sampler.indices[idx][3])

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'actions': torch.from_numpy(data['actions'].astype(np.float32)),
            'cumact_offset': torch.from_numpy(cumact_offset),
            'pad_left': torch.tensor(pad_left, dtype=torch.long),
            'pad_right': torch.tensor(pad_right, dtype=torch.long)
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None, store_image_obs=True,
        repeat_single_frame_image_obs=False):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    depth_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
        elif type == 'depth':
            depth_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['actions'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'actions':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'actions':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['actions']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx, target_h=None, target_w=None):
            try:
                frame = hdf5_arr[hdf5_idx]
                if target_h is not None and target_w is not None:
                    src_h, src_w = frame.shape[:2]
                    if src_h != target_h or src_w != target_w:
                        frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                zarr_arr[zarr_idx] = frame
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        if store_image_obs:
            with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
                # one chunk per thread, therefore no synchronization needed
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                    futures = set()
                    for key in rgb_keys:
                        data_key = 'obs/' + key
                        shape = tuple(shape_meta['obs'][key]['shape'])
                        c,h,w = shape
                        this_compressor = Jpeg2k(level=50)
                        img_arr = data_group.require_dataset(
                            name=key,
                            shape=(n_steps,h,w,c),
                            chunks=(1,h,w,c),
                            compressor=this_compressor,
                            dtype=np.uint8
                        )
                        for episode_idx in range(len(demos)):
                            demo = demos[f'demo_{episode_idx}']
                            hdf5_arr = demo['obs'][key]
                            episode_length = episode_ends[episode_idx] - episode_starts[episode_idx]
                            n_frames = hdf5_arr.shape[0]
                            if n_frames == 1 and repeat_single_frame_image_obs:
                                source_indices = [0] * episode_length
                            elif n_frames == episode_length:
                                source_indices = range(episode_length)
                            else:
                                raise ValueError(
                                    f"data/demo_{episode_idx}/obs/{key} has {n_frames} frames, "
                                    f"but actions has {episode_length}. "
                                    "Expected either 1 frame or one frame per action step."
                                )
                            for step_idx, hdf5_idx in enumerate(source_indices):
                                if len(futures) >= max_inflight_tasks:
                                    # limit number of inflight tasks
                                    completed, futures = concurrent.futures.wait(futures, 
                                        return_when=concurrent.futures.FIRST_COMPLETED)
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError('Failed to encode image!')
                                    pbar.update(len(completed))

                                zarr_idx = episode_starts[episode_idx] + step_idx
                                futures.add(
                                    executor.submit(img_copy,
                                        img_arr, zarr_idx, hdf5_arr, hdf5_idx,
                                        target_h=h, target_w=w))
                    completed, futures = concurrent.futures.wait(futures)
                    for f in completed:
                        if not f.result():
                            raise RuntimeError('Failed to encode image!')
                    pbar.update(len(completed))

            # save depth data (float32, no compression), frame by frame to save memory
            for key in depth_keys:
                data_key = 'obs/' + key
                shape = tuple(shape_meta['obs'][key]['shape'])
                c, h, w = shape
                # check if resize is needed by sampling the first frame
                sample_frame = demos['demo_0'][data_key][0]
                src_h, src_w = sample_frame.shape[:2]
                need_resize = (src_h != h) or (src_w != w)
                # pre-allocate zarr array
                depth_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=None,
                    dtype=np.float32
                )
                # load frame by frame with per-frame progress bar
                with tqdm(total=n_steps, desc=f"Loading depth data ({key})", mininterval=1.0) as pbar:
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo[data_key]
                        episode_length = episode_ends[episode_idx] - episode_starts[episode_idx]
                        n_frames = hdf5_arr.shape[0]
                        if n_frames == 1 and repeat_single_frame_image_obs:
                            source_indices = [0] * episode_length
                        elif n_frames == episode_length:
                            source_indices = range(episode_length)
                        else:
                            raise ValueError(
                                f"data/demo_{episode_idx}/obs/{key} has {n_frames} frames, "
                                f"but actions has {episode_length}. "
                                "Expected either 1 frame or one frame per action step."
                            )

                        for step_idx, hdf5_idx in enumerate(source_indices):
                            frame = hdf5_arr[hdf5_idx].astype(np.float32)
                            # handle (H, W) -> (H, W, 1)
                            if frame.ndim == 2:
                                frame = frame[:, :, np.newaxis]
                            if need_resize:
                                # squeeze channel dim for cv2.resize, then restore
                                if frame.shape[-1] == 1:
                                    frame = frame[:, :, 0]
                                frame = cv2.resize(frame, (w, h),
                                    interpolation=cv2.INTER_NEAREST)
                                if frame.ndim == 2:
                                    frame = frame[:, :, np.newaxis]
                            zarr_idx = episode_starts[episode_idx] + step_idx
                            depth_arr[zarr_idx] = frame
                            pbar.update(1)

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
