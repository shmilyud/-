"""
measure_strawberry_mm_area_updated.py

改进点：
- 三种面积分割方式: 'color' | 'height' | 'hybrid'（默认 hybrid）
- 返回凸包面积 & 栅格化面积（若安装 scipy.ndimage 会做闭运算+填洞）
- 保持全部单位为毫米 (mm)
- 不使用 Open3D，使用 plyfile,numpy,matplotlib,scipy
"""

import os
import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull

# ---------- 配置 ----------
PLY_PATH = r"G:\xiangjicaiji\pointcloud\1765250\RGBDPoints_20240921120714.ply"  # 修改为你的路径
OUT_DIR = os.path.abspath("pcd_measure_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CROP_RADIUS_MM = 150.0
Z_REMOVE_THRESH_MM = 600.0

BLACK_THRESH_CANDIDATES = [50, 80, 100, 150]
GREEN_DELTA_CAND = [30, 20, 10, 5]

POINT_SIZE = 1

# 面积方法: 'color' | 'height' | 'hybrid'
AREA_METHOD = 'hybrid'


# 栅格化参数
MAX_PIXELS_PER_DIM = 2000  # 防止 OOM
MIN_PIXEL_SIZE_MM = 1.0    # 最小像素边长（mm）
# ---------------------------

def find_xyz_rgb_fields(dtype_names):
    names_lower = [n.lower() for n in dtype_names]
    def find_one(cands):
        for c in cands:
            if c in names_lower:
                return dtype_names[names_lower.index(c)]
        return None
    x_f = find_one(['x'])
    y_f = find_one(['y'])
    z_f = find_one(['z'])
    r_f = find_one(['red','r'])
    g_f = find_one(['green','g'])
    b_f = find_one(['blue','b'])
    return x_f, y_f, z_f, r_f, g_f, b_f

def read_ply_points(ply_path):
    plydata = PlyData.read(ply_path)
    el_found = None
    field_names = None
    for elem in plydata.elements:
        names = elem.data.dtype.names
        if names is None:
            continue
        lower = [n.lower() for n in names]
        if ('x' in lower) and ('y' in lower) and ('z' in lower):
            el_found = elem
            field_names = names
            break
    if el_found is None:
        raise ValueError("PLY 文件没有包含 x,y,z 的 element")
    x_f, y_f, z_f, r_f, g_f, b_f = find_xyz_rgb_fields(field_names)
    if x_f is None or y_f is None or z_f is None:
        raise ValueError("无法找到 x/y/z 字段")
    arr = el_found.data
    x = np.asarray(arr[x_f], dtype=np.float64)
    y = np.asarray(arr[y_f], dtype=np.float64)
    z = np.asarray(arr[z_f], dtype=np.float64)
    if r_f is None or g_f is None or b_f is None:
        raise ValueError("PLY 未包含 RGB 字段")
    r = np.asarray(arr[r_f], dtype=np.float64)
    g = np.asarray(arr[g_f], dtype=np.float64)
    b = np.asarray(arr[b_f], dtype=np.float64)
    # 如果颜色是 0-1 范围，扩展到 0-255
    if np.max(r) <= 1.0 and np.max(g) <= 1.0 and np.max(b) <= 1.0:
        r = r * 255.0
        g = g * 255.0
        b = b * 255.0
    pts = np.vstack((x, y, z, r, g, b)).T
    return pts

# 可视化保存函数
def save_xy_scatter(x, y, colors_rgb, title, fname):
    plt.figure(figsize=(6,6))
    cols = np.clip(colors_rgb/255.0, 0.0, 1.0)
    plt.scatter(x, y, c=cols, s=POINT_SIZE)
    plt.axis('equal')
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(title)
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved:", path)

def plot_black_and_all(coords, colors, black_mask, soil_point, fname):
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], c=np.clip(colors/255.0,0,1), s=POINT_SIZE, alpha=0.3)
    if np.any(black_mask):
        plt.scatter(coords[black_mask,0], coords[black_mask,1], c='k', s=POINT_SIZE, label='black candidates')
    if soil_point is not None:
        plt.scatter([soil_point[0]], [soil_point[1]], c='red', s=50, marker='x', label='soil_point')
    plt.axis('equal')
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title("Black soil candidates and chosen soil point")
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200); plt.close(); print("Saved:", path)

def plot_green_and_top(coords, colors, green_mask, plant_top_point, fname):
    plt.figure(figsize=(6,6))
    plt.scatter(coords[:,0], coords[:,1], c=np.clip(colors/255.0,0,1), s=POINT_SIZE, alpha=0.3)
    if np.any(green_mask):
        plt.scatter(coords[green_mask,0], coords[green_mask,1], c='green', s=POINT_SIZE, label='green')
    if plant_top_point is not None:
        plt.scatter([plant_top_point[0]], [plant_top_point[1]], c='blue', s=60, marker='o', label='plant_top')
    plt.axis('equal')
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title("Green points and plant top")
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200); plt.close(); print("Saved:", path)

def plot_projection_with_hull(xy_pts, hull_indices, area_mm2, fname):
    plt.figure(figsize=(6,6))
    plt.scatter(xy_pts[:,0], xy_pts[:,1], c='green', s=POINT_SIZE)
    if hull_indices is not None:
        hull_pts = xy_pts[hull_indices]
        hull_closed = np.vstack((hull_pts, hull_pts[0]))
        plt.plot(hull_closed[:,0], hull_closed[:,1], 'r-', linewidth=1.0)
    plt.axis('equal')
    plt.xlabel("X (mm)"); plt.ylabel("Y (mm)")
    plt.title(f"Projection & Hull  area={area_mm2:.1f} mm^2 ({area_mm2/1e6:.6f} m^2)")
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200); plt.close(); print("Saved:", path)

def plot_3d(points_xyz, colors_rgb, soil_z, soil_point, plant_top_point, fname):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    cols = np.clip(colors_rgb / 255.0, 0.0, 1.0)
    ax.scatter(points_xyz[:,0], points_xyz[:,1], points_xyz[:,2], c=cols, s=POINT_SIZE, alpha=0.6)
    xmin, xmax = np.min(points_xyz[:,0]), np.max(points_xyz[:,0])
    ymin, ymax = np.min(points_xyz[:,1]), np.max(points_xyz[:,1])
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 30), np.linspace(ymin, ymax, 30))
    zz = np.full_like(xx, soil_z)
    ax.plot_surface(xx, yy, zz, color='red', alpha=0.3)
    if soil_point is not None:
        ax.scatter([soil_point[0]], [soil_point[1]], [soil_point[2]], color='black', s=40, marker='x', label='soil_point')
    if plant_top_point is not None:
        ax.scatter([plant_top_point[0]], [plant_top_point[1]], [plant_top_point[2]], color='blue', s=40, marker='o', label='plant_top')
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title("3D view with soil plane")
    ax.legend()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=200); plt.close(); print("Saved:", path)

# ---------- 面积计算函数 ----------
def convex_hull_area_mm2(xy):
    if len(xy) < 3:
        return 0.0, None
    hull = ConvexHull(xy)
    hull_pts = xy[hull.vertices]
    # shoelace:
    x = hull_pts[:,0]; y = hull_pts[:,1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area), hull.vertices

def raster_area_mm2(xy, min_pixel_mm=None):
    # 自动计算像素大小，保证最大维度不超 MAX_PIXELS_PER_DIM
    extent = np.max(xy, axis=0) - np.min(xy, axis=0)
    if min_pixel_mm is None:
        pixel_size = max(MIN_PIXEL_SIZE_MM, extent.max() / float(MAX_PIXELS_PER_DIM))
    else:
        pixel_size = max(min_pixel_mm, extent.max() / float(MAX_PIXELS_PER_DIM))
    grid_w = int(np.ceil(extent[0] / pixel_size)) + 3
    grid_h = int(np.ceil(extent[1] / pixel_size)) + 3
    # 构建栅格
    min_xy = np.min(xy, axis=0)
    indices_x = np.clip(((xy[:,0] - min_xy[0]) / pixel_size).astype(int), 0, grid_w-1)
    indices_y = np.clip(((xy[:,1] - min_xy[1]) / pixel_size).astype(int), 0, grid_h-1)
    grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    grid[indices_y, indices_x] = 1
    # optional: fill holes if scipy.ndimage available
    try:
        from scipy import ndimage
        grid = ndimage.binary_closing(grid, iterations=2).astype(np.uint8)
        grid = ndimage.binary_fill_holes(grid).astype(np.uint8)
    except Exception:
        pass
    count = int(grid.sum())
    area = count * (pixel_size ** 2)  # mm^2
    return float(area), grid, pixel_size, min_xy

# ---------- 主流程 ----------
def main():
    print("=== Load PLY ===")
    pts_all = read_ply_points(PLY_PATH)
    print(f"原始点数: {pts_all.shape[0]}")
    print(f"Z range (mm): min={np.min(pts_all[:,2]):.2f}, max={np.max(pts_all[:,2]):.2f}")

    coords_all = pts_all[:, :3]
    colors_all = pts_all[:, 3:6]

    # 1) 圆形裁剪
    cx, cy = np.mean(coords_all[:,0]), np.mean(coords_all[:,1])
    d2 = (coords_all[:,0]-cx)**2 + (coords_all[:,1]-cy)**2
    mask_circle = d2 <= (CROP_RADIUS_MM**2)
    coords_crop = coords_all[mask_circle]; colors_crop = colors_all[mask_circle]
    print(f"裁剪中心 (X,Y) = ({cx:.2f}, {cy:.2f}) mm")
    print(f"裁剪后点数: {coords_crop.shape[0]}")
    save_xy_scatter(coords_crop[:,0], coords_crop[:,1], colors_crop, "Step1: Cropped (0.2m radius)", "step1_cropped_xy.png")

    # 2) 去除 Z > Z_REMOVE_THRESH_MM
    mask_z = coords_crop[:,2] <= Z_REMOVE_THRESH_MM
    coords_zf = coords_crop[mask_z]; colors_zf = colors_crop[mask_z]
    print(f"去除 Z > {Z_REMOVE_THRESH_MM} mm 后点数: {coords_zf.shape[0]}")
    save_xy_scatter(coords_zf[:,0], coords_zf[:,1], colors_zf, f"Step2: After Z<={Z_REMOVE_THRESH_MM} mm", "step2_after_zfilter_xy.png")
    if coords_zf.shape[0] == 0:
        raise RuntimeError("筛选后无点，请检查阈值或单位")

    # 3) 找黑色土壤点（按候选阈值）
    black_mask = None; used_black_thresh = None
    for bt in BLACK_THRESH_CANDIDATES:
        bm = (colors_zf[:,0] <= bt) & (colors_zf[:,1] <= bt) & (colors_zf[:,2] <= bt)
        if np.sum(bm) > 0:
            black_mask = bm; used_black_thresh = bt; break
    if black_mask is None:
        soil_z = float(np.max(coords_zf[:,2]))
        soil_point = None
        print("未找到黑色点，使用 fallback soil_z = max(Z) =", soil_z)
    else:
        black_pts = coords_zf[black_mask]
        soil_idx = int(np.argmax(black_pts[:,2]))
        soil_point = black_pts[soil_idx]; soil_z = float(soil_point[2])
        print(f"使用黑色阈值 {used_black_thresh} 找到黑点: {np.sum(black_mask)}; soil_z = {soil_z:.2f} mm")
        plot_black_and_all(coords_zf, colors_zf, black_mask, soil_point, "step3_soil_points_xy.png")

    # 4) 找绿色点（按候选 delta）
    green_mask = None; used_green_delta = None
    for gd in GREEN_DELTA_CAND:
        gm = (colors_zf[:,1] > colors_zf[:,0] + gd) & (colors_zf[:,1] > colors_zf[:,2] + gd)
        if np.sum(gm) > 0:
            green_mask = gm; used_green_delta = gd; break
    if green_mask is None:
        print("未找到绿色点（按候选阈值），后续可用 'height' 方式计算面积")
        green_count = 0
    else:
        green_count = int(np.sum(green_mask))
        print(f"使用绿色阈值 delta={used_green_delta} 找到绿色点数: {green_count}")

    # 找植株顶部 Z（如果有绿色点）
    if green_count > 0:
        green_pts = coords_zf[green_mask]
        plant_top_idx = int(np.argmin(green_pts[:,2]))
        plant_top_point = green_pts[plant_top_idx]; plant_top_z = float(plant_top_point[2])
        print(f"植株顶部 Z (min among greens) = {plant_top_z:.2f} mm")
    else:
        # 如果没有绿色点，把植株顶部设为 coords_zf 的最小 Z（防止空）
        plant_top_z = float(np.min(coords_zf[:,2]))
        plant_top_point = None
        print("未检测到绿色点，plant_top_z 设为 coords_zf 最小 Z =", plant_top_z)

    # 5) 株高
    height_mm = soil_z - plant_top_z
    print(f"植株高度 = soil_z - plant_top_z = {height_mm:.2f} mm ({height_mm/1000.0:.4f} m)")

    # 6) 构建用于面积计算的点集（按 AREA_METHOD）
    # black_mask might be None -> create all-false mask
    if black_mask is None:
        black_mask = np.zeros(coords_zf.shape[0], dtype=bool)
    if green_mask is None:
        green_mask = np.zeros(coords_zf.shape[0], dtype=bool)

    if AREA_METHOD == 'color':
        plant_mask = green_mask & (coords_zf[:,2] < soil_z)
        method_note = "color (green mask & Z < soil_z)"
    elif AREA_METHOD == 'height':
        # 选取 soil_z 与 plant_top_z 之间（包含顶部）所有点，排除黑色土壤点
        plant_mask = (coords_zf[:,2] >= plant_top_z) & (coords_zf[:,2] < soil_z) & (~black_mask)
        method_note = "height (Z in [plant_top_z, soil_z) excluding black)"
    else:  # hybrid
        height_mask = (coords_zf[:,2] >= plant_top_z) & (coords_zf[:,2] < soil_z)
        plant_mask = (green_mask | height_mask) & (~black_mask)
        method_note = "hybrid (green OR height) & not black"
    print(f"面积分割方法: {AREA_METHOD} -> {method_note}")
    print(f"选入面积计算的点数: {np.sum(plant_mask)}")

    plant_region_pts = coords_zf[plant_mask]
    area_mm2_convex, hull_idx = 0.0, None
    area_mm2_raster, grid_info = 0.0, None

    if plant_region_pts.shape[0] < 3:
        print("用于面积计算的点不足 (<3)，面积设为 0")
    else:
        # 计算凸包面积
        xy = plant_region_pts[:, :2]
        area_mm2_convex, hull_idx = convex_hull_area_mm2(xy)
        print(f"凸包面积 = {area_mm2_convex:.2f} mm^2 ({area_mm2_convex/1e6:.6f} m^2)")

        # 计算栅格化面积
        area_mm2_raster, grid, pixel_size, min_xy = raster_area_mm2(xy, min_pixel_mm=None)
        print(f"栅格化面积 = {area_mm2_raster:.2f} mm^2  (pixel_size={pixel_size:.3f} mm)")

        # 保存投影与凸包图 & raster 图
        plot_projection_with_hull(xy, hull_idx, area_mm2_convex, "step_projection_hull.png")
        # raster image
        plt.figure(figsize=(6,6))
        plt.imshow(grid, origin='lower')
        plt.title(f"Raster projection (pixel={pixel_size:.3f} mm) area={area_mm2_raster:.1f} mm^2")
        path_r = os.path.join(OUT_DIR, "step_projection_raster.png")
        plt.savefig(path_r, dpi=200); plt.close(); print("Saved:", path_r)

    # 7) 保存 3D 视图
    plot_3d(coords_zf, colors_zf, soil_z, (soil_point.tolist() if soil_point is not None else None), (plant_top_point.tolist() if plant_top_point is not None else None), "step_3d_with_plane_updated.png")

    # 8) 总结打印
    print("\n=== Summary ===")
    print(f"原始点数: {pts_all.shape[0]}")
    print(f"裁剪后点数: {coords_crop.shape[0]}")
    print(f"去除 Z > {Z_REMOVE_THRESH_MM} mm 后点数: {coords_zf.shape[0]}")
    if used_black_thresh is not None:
        print(f"黑色阈值(used)={used_black_thresh}, 黑点数={int(np.sum(black_mask))}")
    else:
        print("未检测到黑色点，使用 fallback soil_z")
    print(f"soil_z (mm) = {soil_z:.2f}")
    print(f"plant_top_z (mm) = {plant_top_z:.2f}")
    print(f"plant_height (mm) = {height_mm:.2f}  ({height_mm/1000.0:.4f} m)")
    print(f"area_convex (mm^2) = {area_mm2_convex:.2f} ; (m^2) = {area_mm2_convex/1e6:.6f}")
    print(f"area_raster (mm^2) = {area_mm2_raster:.2f} ; (m^2) = {area_mm2_raster/1e6:.6f}")
    print("中间图像保存在:", OUT_DIR)
    print("如果面积仍然不对，可以尝试：\n - 改变 AREA_METHOD = 'height' 或 'color'\n - 调整 GREEN_DELTA_CAND 或 BLACK_THRESH_CANDIDATES\n - 调整栅格像素大小 MIN_PIXEL_SIZE_MM（增大可以填补稀疏点）")

if __name__ == "__main__":
    main()
