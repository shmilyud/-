import numpy as np
import cv2
import matplotlib.pyplot as plt
from plyfile import PlyData

# ========== 第一步：读取PLY ==========
def read_ply(filename):
    ply = PlyData.read(filename)
    x = np.array(ply['vertex']['x'])
    y = np.array(ply['vertex']['y'])
    z = np.array(ply['vertex']['z'])
    r = np.array(ply['vertex']['red'])
    g = np.array(ply['vertex']['green'])
    b = np.array(ply['vertex']['blue'])
    return np.vstack((x, y, z)).T, np.vstack((r, g, b)).T

# ========== 第二步：裁剪 ==========
def crop_pointcloud(points, colors, x_range, y_range, z_range):
    mask = (
        (points[:,0] >= x_range[0]) & (points[:,0] <= x_range[1]) &
        (points[:,1] >= y_range[0]) & (points[:,1] <= y_range[1]) &
        (points[:,2] >= z_range[0]) & (points[:,2] <= z_range[1])
    )
    return points[mask], colors[mask]

# ========== 第三步：绿色阈值过滤 ==========
def filter_green(colors, lower, upper):
    hsv = cv2.cvtColor(colors.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask.reshape(-1) > 0

# ========== 第四步：可视化 ==========
def show_points(points, colors, title="PointCloud"):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c=colors/255.0, s=1)
    ax.set_title(title)
    plt.show()

# ========== 第五步：交互调节 ==========
def interactive_green_filter(points, colors):
    def nothing(x): pass

    cv2.namedWindow('Trackbars')

    # HSV绿色范围 (默认大概值)
    cv2.createTrackbar('H_low', 'Trackbars', 35, 180, nothing)
    cv2.createTrackbar('H_high', 'Trackbars', 85, 180, nothing)
    cv2.createTrackbar('S_low', 'Trackbars', 50, 255, nothing)
    cv2.createTrackbar('V_low', 'Trackbars', 50, 255, nothing)

    while True:
        H_low = cv2.getTrackbarPos('H_low', 'Trackbars')
        H_high = cv2.getTrackbarPos('H_high', 'Trackbars')
        S_low = cv2.getTrackbarPos('S_low', 'Trackbars')
        V_low = cv2.getTrackbarPos('V_low', 'Trackbars')

        lower = np.array([H_low, S_low, V_low])
        upper = np.array([H_high, 255, 255])

        mask = filter_green(colors, lower, upper)
        green_points = points[mask]
        green_colors = colors[mask]

        # 实时显示
        if len(green_points) > 0:
            show_points(green_points, green_colors, title=f"H:{H_low}-{H_high} S>{S_low} V>{V_low}")

        key = cv2.waitKey(500) & 0xFF
        if key == 27:  # Esc退出
            break

    cv2.destroyAllWindows()


# ========== 主程序 ==========
if __name__ == "__main__":
    ply_file = r"G:\xiangjicaiji\pointcloud\1765250\RGBDPoints_20240921120714.ply"

    points, colors = read_ply(ply_file)

    # 裁剪，按你需要的范围调整
    points, colors = crop_pointcloud(points, colors,
                                     x_range=(-150, 150),
                                     y_range=(-150, 150),
                                     z_range=(0, 600))


    print("裁剪后点数:", len(points))

    interactive_green_filter(points, colors)
