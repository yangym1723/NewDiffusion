"""
HDF5 图像数据查看器
查看 data/demo_*/obs/camera_depth 和 camera_rgb 的每一帧图片
按键操作:
  A/D  - 上一帧/下一帧
  W/S  - 上一个 demo/下一个 demo
  Q/ESC - 退出
"""

import h5py
import numpy as np
import cv2
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description='HDF5 图像数据查看器')
    parser.add_argument('input', help='HDF5 文件路径')
    args = parser.parse_args()

    f = h5py.File(args.input, 'r')
    data = f['data']

    demo_keys = sorted(
        [k for k in data.keys() if k.startswith('demo_')],
        key=lambda x: int(x.split('_')[1])
    )
    if not demo_keys:
        print("未找到 demo 数据")
        f.close()
        return

    demo_idx = 0
    frame_idx = 0

    def get_demo_info(di):
        dk = demo_keys[di]
        obs = data[dk]['obs']
        has_rgb = 'camera_rgb' in obs
        has_depth = 'camera_depth' in obs
        n_rgb = obs['camera_rgb'].shape[0] if has_rgb else 0
        n_depth = obs['camera_depth'].shape[0] if has_depth else 0
        n_frames = max(n_rgb, n_depth)
        return dk, has_rgb, has_depth, n_rgb, n_depth, n_frames

    def render(di, fi):
        dk, has_rgb, has_depth, n_rgb, n_depth, n_frames = get_demo_info(di)
        if n_frames == 0:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No image data", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            return blank, n_frames

        fi = np.clip(fi, 0, n_frames - 1)
        obs = data[dk]['obs']

        panels = []

        # RGB
        if has_rgb and fi < n_rgb:
            rgb = obs['camera_rgb'][fi]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[-1] == 1:
                rgb = cv2.cvtColor(rgb.squeeze(-1), cv2.COLOR_GRAY2BGR)
            else:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            panels.append(("camera_rgb", rgb))

        # Depth
        if has_depth and fi < n_depth:
            depth = obs['camera_depth'][fi].astype(np.float32)
            if depth.ndim == 3:
                depth = depth.squeeze(-1)
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-7:
                depth_norm = (depth - d_min) / (d_max - d_min)
            else:
                depth_norm = np.zeros_like(depth)
            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            panels.append(("camera_depth", depth_color))

        if not panels:
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Frame {fi} out of range", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            return blank, n_frames

        # 标注
        for label, img in panels:
            cv2.putText(img, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 拼接
        if len(panels) == 2:
            h1, w1 = panels[0][1].shape[:2]
            h2, w2 = panels[1][1].shape[:2]
            if h1 != h2:
                scale = h1 / h2
                panels[1] = (panels[1][0],
                             cv2.resize(panels[1][1], (int(w2 * scale), h1)))
            canvas = np.hstack([p[1] for p in panels])
        else:
            canvas = panels[0][1]

        # 信息栏
        info = f"{dk} [{di+1}/{len(demo_keys)}]  Frame {fi}/{n_frames-1}  |  A/D:frame  W/S:demo  Q:quit"
        bar_h = 40
        bar = np.zeros((bar_h, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        canvas = np.vstack([bar, canvas])

        return canvas, n_frames

    cv2.namedWindow("HDF5 Viewer", cv2.WINDOW_NORMAL)

    while True:
        canvas, n_frames = render(demo_idx, frame_idx)
        frame_idx = np.clip(frame_idx, 0, max(n_frames - 1, 0))
        cv2.imshow("HDF5 Viewer", canvas)

        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), ord('Q'), 27):  # Q / ESC
            break
        elif key in (ord('d'), ord('D')):     # 下一帧
            if frame_idx < n_frames - 1:
                frame_idx += 1
        elif key in (ord('a'), ord('A')):     # 上一帧
            if frame_idx > 0:
                frame_idx -= 1
        elif key in (ord('s'), ord('S')):     # 下一个 demo
            if demo_idx < len(demo_keys) - 1:
                demo_idx += 1
                frame_idx = 0
        elif key in (ord('w'), ord('W')):     # 上一个 demo
            if demo_idx > 0:
                demo_idx -= 1
                frame_idx = 0

    cv2.destroyAllWindows()
    f.close()


if __name__ == '__main__':
    main()
