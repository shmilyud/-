# 全流程（全部单位 mm）
# 步骤：
# 1) 读取 PLY（x,y,z,r,g,b）
# 2) 圆形裁剪（半径 0.2 m = 200 mm，中心为 (x,y) 平均值）
# 3) 去除地面/背景：删除 Z > 600 mm
# 4) 土壤检测：用黑色阈值找黑色点；soil_z_top = 黑色Z的高分位数(默认95%) 作为土壤上表面
#    同时记录 soil_z_max = 黑色Z最大值（仅打印参考）
# 5) 植株顶部：若有绿色点，取绿色点 Z 最小值；否则用全体点 Z 最小值
# 6) 植株高度 = soil_z_top - plant_top_z
# 7) 选取用于面积计算的点：
#    - 方法 'hybrid' (默认): ((green) OR (plant_top_z <= Z < soil_z_top)) AND (非黑色)
#    - 另提供 'color' / 'height' 以便调参
# 8) 计算面积：凸包面积 + 栅格化面积（带闭运算/填洞，若 SciPy 存在）
# 9) 每一步保存并弹窗显示图像（大点云图像作下采样以避免崩溃）
# ------------------------------------------------------------
# 变更点：
#   1) 土壤（黑色）检测仍使用 RGB。
#   2) 绿色植株检测：在 面积方法 AREA_METHOD == 'color' 时，使用 HSV 固定阈值；
#      其余场景（找植株顶部、hybrid 的 green 分支）仍沿用 RGB-delta 逻辑。
# ------------------------------------------------------------

import os
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# --- 仅 color 方法使用的 HSV 阈值（OpenCV 范围：H:0-179, S:0-255, V:0-255） ---
# 你按需修改：
HSV_GREEN_H_LOW   = 36   # 典型绿：约 35~85四个数值由test测试获得
HSV_GREEN_H_HIGH  = 89
HSV_GREEN_S_LOW   = 50  # S、V 提高可过滤暗/灰
HSV_GREEN_V_LOW   = 85

# ========== 配置 ==========
# 你的 PLY 路径
PLY_PATH = r"G:\xiangjicaiji\pointcloud\1765250\RGBDPoints_20240921120714.ply"
OUT_DIR = os.path.abspath("pcd_measure_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# 固定阈值（单位 mm）
CROP_RADIUS_MM = 150.0      # 裁剪半径
Z_REMOVE_THRESH_MM = 600.0  # 俯视图：去掉 Z > 600 mm（地面更远）

# 颜色阈值（黑色 & RGB-绿色）
BLACK_THRESH_CANDIDATES = [50, 80, 100, 150]   # 黑色：RGB 各通道 <= 阈值
GREEN_DELTA_CAND = [30, 20, 10, 5]            # RGB 绿色：G > R + delta 且 G > B + delta（用于找顶部 & hybrid）

# 土壤上表面分位数（避免单点噪声）
SOIL_PLANE_PERCENTILE = 95.0  # 黑色点的 Z 的 95% 分位

# 面积方法: 'color' | 'height' | 'hybrid'
#   - 'color'  ：只用 HSV 绿色点，且 Z < soil_z_top，且非黑色
#   - 'height' ：仅用高度窗 plant_top_z <= Z < soil_z_top 且非黑色
#   - 'hybrid' ：(RGB-green 或 高度窗) 且非黑色
AREA_METHOD = 'color'

# 可视化控制（避免窗口崩溃）
POINT_SIZE = 1
MAX_SHOW_2D = 180_000   # 2D显示最多点数
MAX_SHOW_3D = 60_000    # 3D显示最多点数

# 栅格化设置
MAX_PIXELS_PER_DIM = 2000
MIN_PIXEL_SIZE_MM = 1.0
# ========================


# ---------------- 工具函数 ----------------
def ensure_numpy_rgb_255(r, g, b):
    r = np.asarray(r, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if r.max() <= 1.0 and g.max() <= 1.0 and b.max() <= 1.0:
        r, g, b = r * 255.0, g * 255.0, b * 255.0
    return r, g, b


def find_xyz_rgb_fields(dtype_names):
    names_lower = [n.lower() for n in dtype_names]
    def pick(cands):
        for c in cands:
            if c in names_lower:
                return dtype_names[names_lower.index(c)]
        return None
    return (
        pick(['x']), pick(['y']), pick(['z']),
        pick(['red', 'r']), pick(['green', 'g']), pick(['blue', 'b'])
    )


def read_ply_points(ply_path):
    print("=== 读取 PLY ===")
    plydata = PlyData.read(ply_path)
    el_found, field_names = None, None
    for elem in plydata.elements:
        names = elem.data.dtype.names
        if not names:
            continue
        lower = [n.lower() for n in names]
        if 'x' in lower and 'y' in lower and 'z' in lower:
            el_found, field_names = elem, names
            break
    if el_found is None:
        raise ValueError("PLY 文件未找到含 x,y,z 的 element")

    x_f, y_f, z_f, r_f, g_f, b_f = find_xyz_rgb_fields(field_names)
    if not (x_f and y_f and z_f):
        raise ValueError("PLY 缺少 x/y/z 字段")
    if not (r_f and g_f and b_f):
        raise ValueError("PLY 缺少 RGB 字段")

    arr = el_found.data
    x = np.asarray(arr[x_f], dtype=np.float64)
    y = np.asarray(arr[y_f], dtype=np.float64)
    z = np.asarray(arr[z_f], dtype=np.float64)
    r = np.asarray(arr[r_f], dtype=np.float64)
    g = np.asarray(arr[g_f], dtype=np.float64)
    b = np.asarray(arr[b_f], dtype=np.float64)
    r, g, b = ensure_numpy_rgb_255(r, g, b)

    pts = np.vstack((x, y, z, r, g, b)).T
    print(f"原始点数: {pts.shape[0]}")
    print(f"原始 Z 范围 (mm): min={pts[:,2].min():.2f}, max={pts[:,2].max():.2f}")
    return pts


def downsample_idx(n, k):
    if n <= k:
        return np.arange(n)
    return np.random.choice(n, size=k, replace=False)


def save_and_show(fig, filename, announce=True):
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    if announce:
        print("Saved:", path)
    plt.show()


def scatter_xy(points_xy, colors_rgb, title, filename, max_show=MAX_SHOW_2D):
    n = points_xy.shape[0]
    idx = downsample_idx(n, max_show)
    pts = points_xy[idx]
    cols = np.clip(colors_rgb[idx] / 255.0, 0, 1)

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(pts[:, 0], pts[:, 1], c=cols, s=POINT_SIZE)
    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(title)
    save_and_show(fig, filename)


def plot_black_candidates_all(coords, colors, black_mask, soil_z_top, soil_z_max, filename):
    fig = plt.figure(figsize=(6, 6))
    idx = downsample_idx(coords.shape[0], MAX_SHOW_2D)
    plt.scatter(coords[idx, 0], coords[idx, 1],
                c=np.clip(colors[idx] / 255.0, 0, 1), s=POINT_SIZE, alpha=0.25, label='all')
    if np.any(black_mask):
        bm = np.where(black_mask)[0]
        idx2 = downsample_idx(bm.size, MAX_SHOW_2D // 2)
        bsel = bm[idx2]
        plt.scatter(coords[bsel, 0], coords[bsel, 1], c='k', s=POINT_SIZE, label='black')
    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(f"Black candidates | soil_z_top={soil_z_top:.1f} | soil_z_max={soil_z_max:.1f} (mm)")
    plt.legend(loc='best')
    save_and_show(fig, filename)


def plot_green_and_top(coords, colors, green_mask_rgb, plant_top_point, filename):
    fig = plt.figure(figsize=(6, 6))
    idx = downsample_idx(coords.shape[0], MAX_SHOW_2D)
    plt.scatter(coords[idx, 0], coords[idx, 1],
                c=np.clip(colors[idx] / 255.0, 0, 1), s=POINT_SIZE, alpha=0.25, label='all')

    if np.any(green_mask_rgb):
        gm = np.where(green_mask_rgb)[0]
        idx2 = downsample_idx(gm.size, MAX_SHOW_2D // 2)
        gsel = gm[idx2]
        plt.scatter(coords[gsel, 0], coords[gsel, 1], c='g', s=POINT_SIZE, label='green(RGB)')

    if plant_top_point is not None:
        plt.scatter([plant_top_point[0]], [plant_top_point[1]], c='b', s=60, marker='o', label='plant_top')

    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title("Greens(RGB) & plant top (XY)")
    plt.legend(loc='best')
    save_and_show(fig, filename)


def plot_projection_with_hull(xy_pts, hull_indices, area_mm2, filename):
    fig = plt.figure(figsize=(6, 6))
    idx = downsample_idx(xy_pts.shape[0], MAX_SHOW_2D)
    show_pts = xy_pts[idx]
    plt.scatter(show_pts[:, 0], show_pts[:, 1], c='g', s=POINT_SIZE, alpha=0.6)
    if hull_indices is not None:
        hull_pts = xy_pts[hull_indices]
        hull_closed = np.vstack((hull_pts, hull_pts[0]))
        plt.plot(hull_closed[:, 0], hull_closed[:, 1], 'r-', linewidth=1.2)
    plt.axis('equal'); plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(f"Projection & Convex Hull | area={area_mm2:.1f} mm^2")
    save_and_show(fig, filename)


def plot_raster_grid(grid, pixel_size_mm, area_mm2, filename):
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(grid, origin='lower')
    plt.title(f"Raster proj | pixel={pixel_size_mm:.2f} mm | area={area_mm2:.1f} mm^2")
    save_and_show(fig, filename)


def plot_3d_with_plane(points_xyz, colors_rgb, soil_z_top, filename):
    # 3D点太多会崩，做下采样
    idx = downsample_idx(points_xyz.shape[0], MAX_SHOW_3D)
    pts = points_xyz[idx]; cols = np.clip(colors_rgb[idx] / 255.0, 0, 1)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=POINT_SIZE, alpha=0.65)

    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30),
                         np.linspace(ymin, ymax, 30))
    zz = np.full_like(xx, soil_z_top)
    ax.plot_surface(xx, yy, zz, color='red', alpha=0.3)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title(f"3D view with soil plane (Z={soil_z_top:.1f} mm)")
    save_and_show(fig, filename)


# --------------- 面积计算 ----------------
def convex_hull_area_mm2(xy):
    if xy.shape[0] < 3:
        return 0.0, None
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    x, y = hull_pts[:, 0], hull_pts[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area), hull.vertices


def raster_area_mm2(xy, min_pixel_mm=None):
    extent = xy.max(axis=0) - xy.min(axis=0)
    if min_pixel_mm is None:
        pixel = max(MIN_PIXEL_SIZE_MM, extent.max() / float(MAX_PIXELS_PER_DIM))
    else:
        pixel = max(min_pixel_mm, extent.max() / float(MAX_PIXELS_PER_DIM))

    grid_w = int(np.ceil(extent[0] / pixel)) + 3
    grid_h = int(np.ceil(extent[1] / pixel)) + 3
    min_xy = xy.min(axis=0)

    u = np.clip(((xy[:, 0] - min_xy[0]) / pixel).astype(int), 0, grid_w - 1)
    v = np.clip(((xy[:, 1] - min_xy[1]) / pixel).astype(int), 0, grid_h - 1)

    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    grid[v, u] = 1

    try:
        from scipy import ndimage
        grid = ndimage.binary_closing(grid, iterations=2).astype(np.uint8)
        grid = ndimage.binary_fill_holes(grid).astype(np.uint8)
    except Exception:
        pass

    area = float(grid.sum()) * (pixel ** 2)
    return area, grid, pixel


# ---------------- 主流程 ----------------
def main():
    pts_all = read_ply_points(PLY_PATH)
    coords = pts_all[:, :3]
    colors = pts_all[:, 3:6]
    scatter_xy(coords[:, :2], colors, "Raw XY (subsampled)", "step_raw_xy.png")

    # 1) 圆形裁剪（中心为 XY 均值，半径 CROP_RADIUS_MM）
    cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
    d2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
    mask_circle = d2 <= (CROP_RADIUS_MM ** 2)
    coords_c = coords[mask_circle]
    colors_c = colors[mask_circle]
    print(f"裁剪中心 (X,Y) = ({cx:.2f}, {cy:.2f}) mm")
    print(f"裁剪后点数: {coords_c.shape[0]}")
    scatter_xy(coords_c[:, :2], colors_c, f"After circle crop (R={CROP_RADIUS_MM} mm)", "step1_cropped_xy.png")

    # 2) 去除 Z > Z_REMOVE_THRESH_MM（俯视视角：Z 大为远）
    mask_z = coords_c[:, 2] <= Z_REMOVE_THRESH_MM
    coords_z = coords_c[mask_z]
    colors_z = colors_c[mask_z]
    print(f"去除 Z > {Z_REMOVE_THRESH_MM} mm 后保留点数: {coords_z.shape[0]}")
    scatter_xy(coords_z[:, :2], colors_z, f"Z <= {Z_REMOVE_THRESH_MM} mm", "step2_after_zfilter_xy.png")
    if coords_z.shape[0] == 0:
        raise RuntimeError("Z 阈值过严，筛选后无点。请调小 Z_REMOVE_THRESH_MM 或检查单位。")

    # 3) 黑色土壤点（RGB 阈值）
    used_black_thresh = None
    black_mask = np.zeros(coords_z.shape[0], dtype=bool)
    for bt in BLACK_THRESH_CANDIDATES:
        cand = (colors_z[:, 0] <= bt) & (colors_z[:, 1] <= bt) & (colors_z[:, 2] <= bt)
        if cand.sum() > 0:
            used_black_thresh = bt
            black_mask = cand
            break

    if used_black_thresh is None:
        soil_z_max = float(coords_z[:, 2].max())
        soil_z_top = soil_z_max
        print("未检测到黑色点，fallback：soil_z_top = max(Z)")
    else:
        z_black = coords_z[black_mask, 2]
        soil_z_max = float(z_black.max())
        soil_z_top = float(np.percentile(z_black, SOIL_PLANE_PERCENTILE))
        print(f"黑色阈值(used): {used_black_thresh}；黑色点数: {black_mask.sum()}")
        print(f"soil_z_max(黑色最大Z) = {soil_z_max:.2f} mm")
        print(f"soil_z_top(黑色Z的{SOIL_PLANE_PERCENTILE:.0f}分位) = {soil_z_top:.2f} mm")
        plot_black_candidates_all(coords_z, colors_z, black_mask, soil_z_top, soil_z_max, "step3_black_soil_xy.png")

    # 4) 绿色点（两路：RGB-delta & HSV）
    # 4.1 RGB-delta（用于“找顶部”以及 hybrid 的 green 支路）
    used_green_delta = None
    green_mask_rgb = np.zeros(coords_z.shape[0], dtype=bool)
    for gd in GREEN_DELTA_CAND:
        cand = (colors_z[:, 1] > colors_z[:, 0] + gd) & (colors_z[:, 1] > colors_z[:, 2] + gd)
        if cand.sum() > 0:
            used_green_delta = gd
            green_mask_rgb = cand
            break
    if used_green_delta is None:
        print("未检测到绿色点（RGB-delta）。顶部将用全体点的最小Z。")
    else:
        print(f"绿色(RGB-delta) 阈值 delta={used_green_delta}；绿色点数: {green_mask_rgb.sum()}")

    # 4.2 HSV（仅在 AREA_METHOD == 'color' 时用于面积）
    # OpenCV 的 HSV 需要 BGR 输入
    cols_bgr = colors_z[:, [2, 1, 0]].astype(np.uint8)  # RGB->BGR
    hsv = None
    green_mask_hsv = np.zeros(coords_z.shape[0], dtype=bool)
    if AREA_METHOD == 'color':
        import cv2
        hsv = cv2.cvtColor(cols_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        green_mask_hsv = (h >= HSV_GREEN_H_LOW) & (h <= HSV_GREEN_H_HIGH) & (s >= HSV_GREEN_S_LOW) & (v >= HSV_GREEN_V_LOW)
        print(f"绿色(HSV)  H:[{HSV_GREEN_H_LOW},{HSV_GREEN_H_HIGH}]  S>={HSV_GREEN_S_LOW}  V>={HSV_GREEN_V_LOW} -> 点数: {int(green_mask_hsv.sum())}")

    # 植株顶部（Z 最小）
    if used_green_delta is not None and green_mask_rgb.any():
        plant_top_z = float(coords_z[green_mask_rgb, 2].min())
        plant_top_point = coords_z[green_mask_rgb][np.argmin(coords_z[green_mask_rgb, 2])]
    else:
        plant_top_z = float(coords_z[:, 2].min())
        plant_top_point = coords_z[np.argmin(coords_z[:, 2])]
    print(f"植株顶部 Z = {plant_top_z:.2f} mm")
    plot_green_and_top(coords_z, colors_z, green_mask_rgb, plant_top_point, "step4_greenRGB_and_top_xy.png")

    # 5) 植株高度
    height_mm = soil_z_top - plant_top_z
    print(f"植株高度 = soil_z_top - plant_top_z = {height_mm:.2f} mm ({height_mm/1000.0:.4f} m)")

    # 6) 面积分割
    if AREA_METHOD == 'color':
        # 仅使用 HSV 的绿色阈值作为“颜色”分割，并与高度方向一致（Z < soil_z_top），同时剔除黑色
        plant_mask = green_mask_hsv & (coords_z[:, 2] < soil_z_top) & (~black_mask)
        method_note = f"color(HSV): H[{HSV_GREEN_H_LOW},{HSV_GREEN_H_HIGH}], S>={HSV_GREEN_S_LOW}, V>={HSV_GREEN_V_LOW}, 且 Z<soil_z_top, 非黑色"
    elif AREA_METHOD == 'height':
        # 纯高度窗 + 非黑色
        plant_mask = (coords_z[:, 2] >= plant_top_z) & (coords_z[:, 2] < soil_z_top) & (~black_mask)
        method_note = "height: plant_top_z <= Z < soil_z_top (excl. black)"
    else:  # 'hybrid'
        height_mask = (coords_z[:, 2] >= plant_top_z) & (coords_z[:, 2] < soil_z_top)
        plant_mask = (green_mask_rgb | height_mask) & (~black_mask)
        method_note = "hybrid: (green(RGB-delta) OR height) AND not black"

    print(f"面积分割方法: {AREA_METHOD} -> {method_note}")
    print(f"选入面积计算点数: {plant_mask.sum()}")

    # 7) 面积计算
    area_convex_mm2, hull_idx = 0.0, None
    area_raster_mm2, raster_grid, pixel_mm = 0.0, None, None

    plant_xy = coords_z[plant_mask, :2]
    if plant_xy.shape[0] < 3:
        print("用于面积计算的点不足 (<3)。面积置 0。")
    else:
        area_convex_mm2, hull_idx = convex_hull_area_mm2(plant_xy)
        print(f"凸包面积 = {area_convex_mm2:.2f} mm^2 ({area_convex_mm2/1e6:.6f} m^2)")
        plot_projection_with_hull(plant_xy, hull_idx, area_convex_mm2, "step5_projection_hull.png")

        area_raster_mm2, raster_grid, pixel_mm = raster_area_mm2(plant_xy, min_pixel_mm=None)
        print(f"栅格化面积 = {area_raster_mm2:.2f} mm^2  (pixel={pixel_mm:.2f} mm)")
        plot_raster_grid(raster_grid, pixel_mm, area_raster_mm2, "step6_projection_raster.png")

    # 8) 3D 视图（带土壤上表面平面），**弹窗显示**
    plot_3d_with_plane(coords_z, colors_z, soil_z_top, "step7_3d_with_soil_plane.png")

    # 9) 汇总
    print("\n=== Summary ===")
    print(f"裁剪后点数: {coords_c.shape[0]}")
    print(f"去除 Z > {Z_REMOVE_THRESH_MM} mm 后点数: {coords_z.shape[0]}")
    if used_black_thresh is not None:
        print(f"黑色阈值(used) = {used_black_thresh}；黑色点数 = {black_mask.sum()}")
        print(f"soil_z_max = {soil_z_max:.2f} mm; soil_z_top(perc {SOIL_PLANE_PERCENTILE:.0f}) = {soil_z_top:.2f} mm")
    else:
        print("未检测到黑色点（使用 fallback soil_z_top = max(Z)）")
    print(f"植株顶部 Z = {plant_top_z:.2f} mm")
    print(f"植株高度 = {height_mm:.2f} mm ({height_mm/1000.0:.4f} m)")
    print(f"area_convex = {area_convex_mm2:.2f} mm^2 ({area_convex_mm2/1e6:.6f} m^2)")
    print(f"area_raster = {area_raster_mm2:.2f} mm^2 ({area_raster_mm2/1e6:.6f} m^2)")
    print("中间图像保存在:", OUT_DIR)
    print("如需调参：\n"
          f"- HSV 阈值：HSV_GREEN_H_LOW/HIGH, HSV_GREEN_S_LOW, HSV_GREEN_V_LOW（仅 color 方法）\n"
          f"- RGB 绿色阈值列表：GREEN_DELTA_CAND（找顶部 & hybrid）\n"
          f"- 黑色阈值列表：BLACK_THRESH_CANDIDATES（土壤）\n"
          f"- SOIL_PLANE_PERCENTILE（上表面分位数）\n"
          f"- CROP_RADIUS_MM, Z_REMOVE_THRESH_MM（几何过滤）")


if __name__ == "__main__":
    main()
