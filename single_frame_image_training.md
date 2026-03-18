# 单帧图像训练改动说明

本文档说明本次为 `train_diffusion_transformer_real_hybrid_workspace` 训练链路所做的修改，以及新增配置的使用方式。

目标是解决以下问题：

- 数据集里的相机观测只想保留单帧图像。
- 训练时模型仍然需要拿到长度为 `n_obs_steps` 的观测序列。
- 旧流程需要先用 `process_hdf5.py` 把单帧图像复制成与 `actions` 等长，耗时且占用大量磁盘空间。
- 现在改为：磁盘上保留单帧，训练时由 dataset 在内存中重复读取这一帧，生成模型所需的多步观测张量。


## 1. 本次修改的核心思路

以前的流程是：

1. 采集数据时保存单帧相机图像。
2. 用 `process_hdf5.py` 将这 1 帧复制为 `T` 帧，`T = actions.shape[0]`。
3. 训练时按原有时序数据格式读取。

现在的流程是：

1. 采集数据时仍然只保存单帧相机图像。
2. 用 `process_hdf5.py` 处理数据时，默认不再扩展相机帧。
3. 训练时如果发现某个相机观测 key 只有 1 帧，就在 `dataset.__getitem__` 中按需要重复这一帧。
4. 模型看到的输入张量形状不变，因此不需要改模型结构和训练命令。

对于 action，还额外支持了一类“episode 内固定维度”：

- 当前配置里是 `persistent_dims: [7, 8]`
- 这表示 action 的第 8、9 维在一局开始后应保持固定
- 模型会在 episode 第一次 query 时预测出这两维
- 后续 rollout 中这两维都会复用第一次预测值
- 这两维也会从 `cumact` 累加里排除，避免被错误地当作时序控制量累加


## 2. 改动了哪些文件

### 2.1 `diffusion_policy/dataset/robomimic_replay_image_dataset.py`

这是本次最核心的修改文件。

新增能力：

- 支持图像观测不写入 zarr cache，而是训练时直接从原始 HDF5 读取。
- 支持当图像序列长度为 1 时，在训练时自动重复该帧。
- 支持 depth 在不展开到磁盘的情况下，直接从 HDF5 统计 normalizer 所需的 min/max/mean/std。

新增的 dataset 参数：

- `load_image_obs_from_hdf5`
  - `False`：保持旧行为，先把图像写入 replay buffer / zarr cache。
  - `True`：图像不写入 replay buffer，训练取样时直接从原始 HDF5 读取。

- `repeat_single_frame_image_obs`
  - `False`：如果某个图像 key 只有 1 帧，会直接报错。
  - `True`：如果某个图像 key 只有 1 帧，dataset 会在训练时自动重复这 1 帧。

- `preload_single_frame_image_obs`
  - `False`：单帧图像第一次被 sample 用到时再从 HDF5 读取。
  - `True`：启动训练时就把每个 episode 的单帧图像预加载到内存，训练时直接从内存重复，速度更快。

- `persistent_dims`
  - 指定哪些 action 维度在一个 episode 内应该保持固定。
  - 当前配置使用 `[7, 8]`，即倒数第三维和倒数第二维。

现在的实现逻辑是：

- `actions` 和 low-dim obs 仍然照常进入 replay buffer。
- RGB / depth 可以不进入 replay buffer。
- 当 `__getitem__` 被调用时：
  - dataset 先根据 sample window 算出当前样本实际需要的时间步。
  - 如果图像序列长度等于 episode length，则按原时序读取。
  - 如果图像序列长度等于 1 且开启了 `repeat_single_frame_image_obs=True`，则直接重复读取索引 `0`。

这意味着：

- 模型输入 shape 不变。
- 磁盘上不再需要保存重复的一千多帧相同图像。
- cache 体积和构建时间会明显下降。


### 2.2 `diffusion_policy/config/task/real_pusht_image.yaml`

这个 task config 已经默认开启了新的训练方式：

```yaml
dataset:
  _target_: diffusion_policy.dataset.robomimic_replay_image_dataset.RobomimicReplayImageDataset
  ...
  use_cache: True
  load_image_obs_from_hdf5: True
  repeat_single_frame_image_obs: True
  preload_single_frame_image_obs: True

actions:
  shape: [10]
  persistent_dims: [7, 8]
  binary_dims: [9]
```

含义如下：

- `use_cache: True`
  - 仍然缓存 low-dim 和 action 数据。

- `load_image_obs_from_hdf5: True`
  - 图像不缓存到 zarr 中，而是在训练时从 HDF5 读取。

- `repeat_single_frame_image_obs: True`
  - 如果某个图像 key 只有一帧，则训练时自动重复使用它。

- `preload_single_frame_image_obs: True`
  - 如果某个图像 key 只有一帧，则在 dataset 初始化时先把这帧读到内存，训练时不再反复访问 HDF5。

- `persistent_dims: [7, 8]`
  - 表示 action 的第 8、9 维在一局内不应变化。
  - rollout 时第一次 query policy 得到这两维后，会缓存下来并复用到整局结束。
  - 训练时也会把这两维视为 episode 级固定量，并从 `cumact` 中排除。


### 2.3 `process_hdf5.py`

这个脚本的默认行为已经修改。

旧行为：

- 遇到 `camera_rgb` / `camera_depth` 只有 1 帧时，自动扩展到与 `actions` 一样长。

新行为：

- 默认不扩展。
- 只有显式传入 `--expand-camera-frames` 时才执行旧行为。

也就是说，现在你即使继续使用这个脚本，默认也不会再把单帧图像复制到上千帧。


### 2.4 `tests/test_robomimic_replay_image_dataset_single_frame.py`

新增了一个最小测试，验证以下行为：

- 当 HDF5 中 `camera_rgb` / `camera_depth` 只有 1 帧时，dataset 可以正常取样。
- dataset 输出的 `obs` 仍然是 `n_obs_steps` 长度。
- 每个时间步拿到的都是同一帧图像。
- 图像 key 不再强制进入 replay buffer。


## 3. 现在的数据格式要求

### 3.1 对 low-dim 和 action 的要求

这些部分和以前一致：

- `data/demo_*/actions` 必须是完整时序，长度为 `T`。
- `data/demo_*/obs` 中的 low-dim 项，例如：
  - `ee_pose`
  - `ee_quat`
  - `contact_force_z`
  - 其他 low-dim key
  仍然应该是长度为 `T` 的完整序列。


### 3.2 对图像观测的要求

对于所有在 `shape_meta.obs` 中声明为以下类型的 key：

- `type: rgb`
- `type: depth`

现在允许两种合法格式：

1. 完整时序
   - 图像长度等于 `actions.shape[0]`
   - 这时会按原始时序读取

2. 单帧
   - 图像长度等于 `1`
   - 这时会在训练时自动重复该帧

如果图像长度既不是 `1`，也不是 `actions.shape[0]`，dataset 会报错。


## 4. 训练时新增配置如何使用

### 4.1 默认情况

你当前的训练命令不需要改：

```bash
python train.py --config-name=train_diffusion_transformer_real_hybrid_workspace
```

因为 `diffusion_policy/config/task/real_pusht_image.yaml` 已经默认开启了：

```yaml
load_image_obs_from_hdf5: True
repeat_single_frame_image_obs: True
```


### 4.2 如果你想关闭这个功能

如果以后你又想恢复旧逻辑，可以把 task config 里的这两个参数改掉：

```yaml
load_image_obs_from_hdf5: False
repeat_single_frame_image_obs: False
```

此时行为会回到旧模式：

- 图像会被转换进 replay buffer / zarr cache。
- 数据集中的图像序列必须是完整长度。
- 单帧图像如果不先扩展，会报错。


### 4.3 配置项详细解释

#### `load_image_obs_from_hdf5`

建议值：`True`

作用：

- 图像在训练时直接从原始 HDF5 读取。
- 避免将图像写入 cache。
- 能显著减少 cache 占用和预处理时间。

注意：

- 训练时会多一些 HDF5 读取开销。
- 但和提前把几百 GB 的重复图像写到磁盘相比，这个代价通常更合理。


#### `repeat_single_frame_image_obs`

建议值：`True`

作用：

- 当某个 RGB / depth key 只有 1 帧时，dataset 自动重复这一帧。

注意：

- 这个逻辑是按 key 生效的。
- 只要该 key 在 `shape_meta.obs` 中被标记为 `rgb` 或 `depth`，就会走这个逻辑。


#### `preload_single_frame_image_obs`

建议值：`True`

作用：

- 将单帧图像在训练开始时预加载到内存。
- 后续每个 sample 直接从内存重复该帧，而不是反复读取 HDF5。
- 在“单帧图像 + 大量训练窗口”的场景下，能明显提升训练速度。

代价：

- 会多占用一些内存。
- 但占用的是“每个 episode 每个图像 key 的一帧”，远小于把整段视频在磁盘上扩展到和 `actions` 等长。


## 5. `process_hdf5.py` 现在怎么用

### 5.1 推荐用法

如果你的原始数据集里相机观测本来就是单帧，推荐直接这样处理：

```bash
python process_hdf5.py your_data.hdf5 -o your_data_processed.hdf5
```

此时：

- `actions` 会被正常复制过去。
- low-dim obs 会被正常复制过去。
- `camera_rgb` / `camera_depth` 如果只有 1 帧，会保持 1 帧，不会展开。


### 5.2 如果你仍然需要兼容旧流程

可以显式加上：

```bash
python process_hdf5.py your_data.hdf5 -o your_data_processed.hdf5 --expand-camera-frames
```

这会恢复旧行为，把单帧 `camera_rgb` / `camera_depth` 扩展到与 `actions` 等长。


## 6. 你的实际推荐工作流

推荐你以后按下面的流程使用：

1. 采集数据时：
   - low-dim / actions 仍然保存完整时序。
   - 图像观测只保存你需要的那一帧。

2. 处理数据时：
   - 运行：
     ```bash
     python process_hdf5.py your_data.hdf5 -o your_data_processed.hdf5
     ```
   - 不要加 `--expand-camera-frames`。

3. 训练时：
   - 直接运行：
     ```bash
     python train.py --config-name=train_diffusion_transformer_real_hybrid_workspace
     ```

4. dataset 行为：
   - 发现图像 key 长度为 1 时，在内存里自动重复。
   - 模型仍然收到符合原先接口的多步观测张量。


## 7. 如果你有多个相机 key，怎么理解这个功能

你提到你的 `obs` 中包含两个相机的数据。

本次实现不是只针对 `camera_rgb` / `camera_depth` 这两个固定名字在训练时硬编码处理，而是按 `shape_meta.obs` 中的类型来分：

- `type: rgb`
- `type: depth`

因此如果你的任务配置里有多个相机 key，例如：

```yaml
shape_meta:
  obs:
    camera_0_rgb:
      shape: [3, 240, 320]
      type: rgb
    camera_1_rgb:
      shape: [3, 240, 320]
      type: rgb
```

只要这些 key 对应的 HDF5 数据满足：

- 要么长度是 `1`
- 要么长度是 `actions.shape[0]`

就都可以使用同样的逻辑。

注意：

- 当前 `process_hdf5.py` 中默认“是否扩展”的判断仍然只写了 `camera_rgb` 和 `camera_depth` 这两个名字。
- 但训练时 dataset 的单帧重复逻辑是对所有 `rgb` / `depth` 类型 key 都生效的。

如果你后续希望 `process_hdf5.py` 也对你自定义的多相机 key 名称做统一处理，可以再继续扩展这个脚本。


## 8. 和旧 cache 的关系

由于这次 dataset 的 cache 指纹里加入了：

- `load_image_obs_from_hdf5`
- `repeat_single_frame_image_obs`

所以当你启用新逻辑时，会自动生成新的 cache 文件，不会误用旧 cache。

但需要注意：

- 旧的 `.zarr.zip` cache 文件不会自动删除。
- 如果你确认不用了，可以手动删除旧 cache，释放磁盘空间。


## 9. 当前方案的优点

- 不再需要把单帧图像扩展成上千帧后再训练。
- 明显减少 `process_hdf5.py` 的处理时间。
- 明显降低数据预处理期间的磁盘占用。
- 不需要改模型结构。
- 不需要改训练命令。
- 模型拿到的 batch shape 与旧流程保持一致。


## 10. 当前方案的限制

- low-dim 序列和 `actions` 仍然必须是完整时序。
- 图像 key 的合法长度目前只支持两种：
  - `1`
  - `T = actions.shape[0]`
- 如果你的图像长度是别的值，例如 `5`、`16`、`100`，当前实现会报错。
- `process_hdf5.py` 的“相机 key 自动识别”目前仍然是按 `camera_rgb` / `camera_depth` 这两个名字写的。


## 11. 你现在最常用的命令

### 数据后处理

```bash
python process_hdf5.py your_data.hdf5 -o your_data_processed.hdf5
```

### 开始训练

```bash
python train.py --config-name=train_diffusion_transformer_real_hybrid_workspace
```

### 如果你确实还想用旧的图像扩展流程

```bash
python process_hdf5.py your_data.hdf5 -o your_data_processed.hdf5 --expand-camera-frames
```


## 12. 总结

本次修改后，你可以保持数据集中的相机观测为单帧，不再需要在磁盘上把这一帧复制成上千帧。训练阶段会由 dataset 自动反复读取这一帧来构造模型所需的多步图像观测，因此能显著减少预处理时间和磁盘占用，同时保持现有训练配置和模型结构基本不变。
