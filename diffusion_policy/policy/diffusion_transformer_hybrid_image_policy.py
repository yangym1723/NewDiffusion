from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # cumulative action history encoder
            use_cumact_encoder=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['actions']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # binary action dims (e.g. gripper on/off) need special handling
        self.binary_action_dims = shape_meta['actions'].get('binary_dims', None)
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            elif type == 'depth':
                obs_config['rgb'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        cumact_input_dim = action_dim if use_cumact_encoder else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers,
            cumact_input_dim=cumact_input_dim
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.use_cumact_encoder = use_cumact_encoder
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            raw_cumact_offset=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler
        use_cumact = (model.cumact_encoder is not None)

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        # initialize cumact
        cumact = None
        if use_cumact:
            # Start with the episode-level offset normalized via cumact normalizer
            if raw_cumact_offset is not None:
                # (B, Da) -> (B, 1, Da) -> broadcast to (B, T, Da)
                raw_offset_expanded = raw_cumact_offset.unsqueeze(1).expand(
                    -1, condition_data.shape[1], -1)
                cumact = self.normalizer['cumact'].normalize(raw_offset_expanded)
            else:
                # No episode offset provided; treat as episode start (raw cumact = 0).
                # Must normalize rather than using literal zeros, because
                # normalize(0) = offset ≠ 0 in general for affine normalizers.
                raw_zero = torch.zeros(
                    condition_data.shape[0], condition_data.shape[1],
                    self.action_dim, device=condition_data.device,
                    dtype=condition_data.dtype)
                cumact = self.normalizer['cumact'].normalize(raw_zero)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond, cumact=cumact)

            # 3. compute previous image: x_t -> x_t-1
            step_output = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                )
            trajectory = step_output.prev_sample

            # 4. update cumact from predicted clean trajectory (x0 estimate)
            #    All computation in RAW action space, then normalize with cumact normalizer.
            if use_cumact:
                pred_x0 = step_output.pred_original_sample
                if pred_x0 is not None:
                    # Unnormalize predicted actions back to raw space
                    pred_x0_act = pred_x0[..., :self.action_dim]
                    raw_pred = self.normalizer['actions'].unnormalize(pred_x0_act)

                    # Exclude binary dims
                    if self.binary_action_dims is not None:
                        raw_pred = raw_pred.clone()
                        for dim in self.binary_action_dims:
                            raw_pred[..., dim] = 0.0

                    # Raw cumsum, right-shifted
                    raw_cs = torch.cumsum(raw_pred, dim=1)
                    raw_intra = torch.cat([
                        torch.zeros_like(raw_cs[:, :1, :]),
                        raw_cs[:, :-1, :]
                    ], dim=1)

                    # Add episode-level raw offset
                    if raw_cumact_offset is not None:
                        raw_full = raw_intra + raw_cumact_offset.unsqueeze(1)
                    else:
                        raw_full = raw_intra

                    # Normalize with cumact normalizer
                    cumact = self.normalizer['cumact'].normalize(raw_full)

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor],
                       episode_cumact: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        episode_cumact: (B, Da) cumulative sum of all previously executed actions
                        in the current episode. None if cumact is not used.
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # normalize episode_cumact if provided
        # episode_cumact is in RAW action space (sum of executed raw actions).
        # We pass it as-is to conditional_sample which will handle
        # the raw-space cumact computation and cumact normalization.
        raw_cumact_offset = None
        if episode_cumact is not None and self.use_cumact_encoder:
            raw_cumact_offset = episode_cumact

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            cond=cond,
            raw_cumact_offset=raw_cumact_offset,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['actions'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'actions': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['actions'].normalize(batch['actions'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # compute episode-level cumulative action history (teacher forcing)
        # All cumact computation is done in RAW (unnormalized) action space to
        # avoid the affine normalizer's non-additive offset issue.  The result
        # is then normalized with a dedicated cumact normalizer fitted on the
        # actual cumulative sum distribution.
        cumact = None
        if self.model.cumact_encoder is not None:
            raw_actions = batch['actions']               # (B, T, Da)  raw
            raw_offset  = batch['cumact_offset']         # (B, Da)     raw

            # Prepare raw actions for cumact: zero binary dims + padded positions
            raw_for_cumact = raw_actions.clone()
            if self.binary_action_dims is not None:
                for dim in self.binary_action_dims:
                    raw_for_cumact[..., dim] = 0.0
            if 'pad_left' in batch:
                for i in range(batch_size):
                    pl = batch['pad_left'][i].item()
                    if pl > 0:
                        raw_for_cumact[i, :pl, :] = 0.0
            if 'pad_right' in batch:
                for i in range(batch_size):
                    pr = batch['pad_right'][i].item()
                    if pr > 0:
                        raw_for_cumact[i, -pr:, :] = 0.0

            # Intra-window cumulative sum, right-shifted
            raw_cs = torch.cumsum(raw_for_cumact, dim=1)
            raw_intra = torch.cat([
                torch.zeros_like(raw_cs[:, :1, :]),
                raw_cs[:, :-1, :]
            ], dim=1)  # (B, T, Da)

            # Full episode-level raw cumact
            raw_full_cumact = raw_intra + raw_offset.unsqueeze(1)  # (B, T, Da)

            # Normalize with dedicated cumact normalizer
            full_cumact = self.normalizer['cumact'].normalize(raw_full_cumact)

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
                if self.model.cumact_encoder is not None:
                    cumact = full_cumact[:, start:end]
            else:
                if self.model.cumact_encoder is not None:
                    cumact = full_cumact
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()
            if self.model.cumact_encoder is not None:
                cumact = full_cumact

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond, cumact=cumact)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
