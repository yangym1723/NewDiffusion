"""
HDF5 数据预处理脚本
- data/demo_*/actions -> data/demo_*/actions (保持原名)
- data/demo_*/obs/actions -> data/demo_*/obs/actions (保持原名)
- data/demo_*/obs/ee_pose 保持原名不变
- data/demo_*/obs/camera_depth 和 camera_rgb: 将单帧图片扩展到与 actions 帧数相同
python process_hdf5.py your_data.hdf5
# 输出: your_data_processed.hdf5

# 或指定输出路径
python process_hdf5.py your_data.hdf5 -o output.hdf5
"""

import h5py
import numpy as np
import argparse
import os


def process_hdf5(input_path, output_path):
    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        # 复制顶层属性
        for attr_key, attr_val in fin.attrs.items():
            fout.attrs[attr_key] = attr_val

        data_group = fin['data']
        out_data = fout.create_group('data')
        for attr_key, attr_val in data_group.attrs.items():
            out_data.attrs[attr_key] = attr_val

        demo_keys = sorted(
            [k for k in data_group.keys() if k.startswith('demo_')],
            key=lambda x: int(x.split('_')[1])
        )

        for demo_key in demo_keys:
            demo_in = data_group[demo_key]
            demo_out = out_data.create_group(demo_key)
            for attr_key, attr_val in demo_in.attrs.items():
                demo_out.attrs[attr_key] = attr_val

            # --- 处理 actions (demo 级别) ---
            if 'actions' in demo_in:
                action_data = demo_in['actions'][:]
                demo_out.create_dataset('actions', data=action_data)
            elif 'action' in demo_in:
                action_data = demo_in['action'][:]
                demo_out.create_dataset('actions', data=action_data)
            else:
                raise KeyError(f"{demo_key} 中找不到 'actions' 或 'action' 数据项")

            T = action_data.shape[0]

            # --- 处理 obs ---
            obs_in = demo_in['obs']
            obs_out = demo_out.create_group('obs')
            for attr_key, attr_val in obs_in.attrs.items():
                obs_out.attrs[attr_key] = attr_val

            for obs_key in obs_in.keys():
                src_data = obs_in[obs_key][:]

                # actions 保持原名
                if obs_key == 'actions':
                    out_key = 'actions'
                # ee_pose 保持原名不变
                elif obs_key == 'ee_pose':
                    out_key = 'ee_pose'
                else:
                    out_key = obs_key

                # 扩展: camera_depth 和 camera_rgb
                if obs_key in ('camera_depth', 'camera_rgb'):
                    if src_data.shape[0] == 1:
                        src_data = np.repeat(src_data, T, axis=0)
                        print(f"  {demo_key}/obs/{obs_key}: 单帧扩展到 {T} 帧 -> obs/{out_key}")
                    elif src_data.shape[0] != T:
                        print(f"  警告: {demo_key}/obs/{obs_key} 帧数={src_data.shape[0]}, "
                              f"action 帧数={T}, 不做扩展")

                obs_out.create_dataset(out_key, data=src_data)

            # --- 复制 demo 下除 actions/action/obs 之外的其他数据项 ---
            for key in demo_in.keys():
                if key in ('actions', 'action', 'obs'):
                    continue
                if isinstance(demo_in[key], h5py.Dataset):
                    demo_out.create_dataset(key, data=demo_in[key][:])
                elif isinstance(demo_in[key], h5py.Group):
                    fin.copy(demo_in[key], demo_out, name=key)

        # --- 复制 data 以外的其他顶层 group/dataset ---
        for key in fin.keys():
            if key == 'data':
                continue
            if isinstance(fin[key], h5py.Dataset):
                fout.create_dataset(key, data=fin[key][:])
            elif isinstance(fin[key], h5py.Group):
                fin.copy(fin[key], fout, name=key)

    print(f"\n处理完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='HDF5 数据预处理脚本')
    parser.add_argument('input', help='输入 HDF5 文件路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出 HDF5 文件路径 (默认: 输入文件名_processed.hdf5)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_processed{ext}"

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        raise ValueError("输入和输出文件路径不能相同")

    print(f"输入: {args.input}")
    print(f"输出: {args.output}")
    print()

    process_hdf5(args.input, args.output)


if __name__ == '__main__':
    main()
