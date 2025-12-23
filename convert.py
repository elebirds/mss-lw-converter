import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
import io

# ==========================================
# 1. 物理常数
# ==========================================
CONSTANTS = {
    'RE': 6371.2 * 1000,  # 地球半径 (m)
    'ME': 8.0e15,  # 地球磁矩 (T·m^3)
    'QE': 1.602e-19,  # 电子电荷 (C)
    'E0_KEV': 511.0,  # 电子静止能量 (keV)
    'KEV_TO_J': 1.602e-16,  # keV 转 Joules
    'W_COR_RAD_H': 2 * np.pi / 24.0  # 地球自转角速度 (rad/hour)
}


# ==========================================
# 2. 数据读取 (保持不变)
# ==========================================
def parse_energy_bins(lines: list[str]) -> np.ndarray:
    try:
        energy_line = next(l for l in lines if "Energy [keV]" in l)
    except StopIteration:
        raise ValueError("未在文件头中找到 'Energy [keV]' 信息")

    content = energy_line.split("Energy [keV]")[-1]
    tokens = content.replace(',', ' ').split()
    boundaries = []
    for tok in tokens:
        try:
            boundaries.append(float(tok))
        except ValueError:
            continue

    boundaries_np = np.array(boundaries, dtype=float)
    # 计算几何中心或算术中心
    return 0.5 * (boundaries_np[:-1] + boundaries_np[1:])


def load_fedo_data_from_lines(lines: list[str]):
    e_centers = parse_energy_bins(lines)

    try:
        header_line = next(l for l in lines if l.strip().startswith("yyyy"))
    except StopIteration:
        raise ValueError("未找到表头")

    header_idx = lines.index(header_line)
    colnames = [c.strip() for c in header_line.split(',') if c.strip()]
    data_str = "".join(lines[header_idx + 1:])
    df = pd.read_csv(io.StringIO(data_str), header=None, names=colnames, sep=',', on_bad_lines='skip')

    # 提取需要的列
    e_cols = [c for c in df.columns if c.startswith('E_') and c[2:].isdigit()]
    min_len = min(len(e_cols), len(e_centers))
    e_cols = e_cols[:min_len]
    e_centers = e_centers[:min_len]

    # 清洗数值
    for col in e_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in [c for c in df.columns if 'POS' in c or 'L_' in c]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 关键修复：在添加 Datetime 新列之前整理内存
    df = df.copy()

    # 时间处理
    df['Datetime'] = pd.to_datetime(df[['yyyy', 'mn', 'dd', 'hh', 'mm', 'ss']].astype(str).agg('-'.join, axis=1), format='%Y-%m-%d-%H-%M-%S')

    return df, e_centers, e_cols


def load_fedo_data(file_path: str):
    with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
        lines = f.readlines()
    return load_fedo_data_from_lines(lines)


# ==========================================
# 3. 物理计算 (核心)
# ==========================================
def calculate_physics_data(df, e_centers, e_cols):
    # 1. 提取基础数据
    try:
        x_col = next(c for c in df.columns if 'POS_GSEx' in c)
        y_col = next(c for c in df.columns if 'POS_GSEy' in c)
        z_col = next(c for c in df.columns if 'POS_GSEz' in c)
        l_col = next(c for c in df.columns if 'L_Dipole' in c)

        r_vec = df[[x_col, y_col, z_col]].values
        l_vals = df[l_col].values
        flux_matrix = df[e_cols].values
        if 'Datetime' in df.columns:
            time_vals = df['Datetime'].values
        else:
            time_vals = np.arange(len(df))
    except StopIteration:
        raise KeyError("关键列缺失")

    # 2. 过滤无效点
    mask = (l_vals >= 1.0) & np.isfinite(l_vals) & (np.linalg.norm(r_vec, axis=1) > 0)
    l_vals = l_vals[mask]
    r_vec = r_vec[mask]
    flux_matrix = flux_matrix[mask]
    time_vals = time_vals[mask]

    # 3. 计算 wd (矩阵化)
    # 3.1 几何因子 F(y)
    r_norm = np.linalg.norm(r_vec, axis=1)
    cos2 = np.clip(r_norm / (l_vals * CONSTANTS['RE']), 0, 1)    # 实际上 = cos^2(λ) （λ 为磁纬）
    sin2 = 1.0 - cos2
    y = ((cos2 ** 3) / np.sqrt(1 + 3 * sin2)) ** 0.5 # y = (cos^6 / sqrt(1 + 3sin^2))^0.5
    y_pow = y ** 0.75
    F_y = (5.520692 - 2.357194 * y + 1.279385 * y_pow) / (12 * (1.380173 - 0.639693 * y_pow))

    # 3.2 能量项
    E_J = e_centers * CONSTANTS['KEV_TO_J']
    E0_J = CONSTANTS['E0_KEV'] * CONSTANTS['KEV_TO_J']
    gamma_term = (E_J * (E_J + 2 * E0_J)) / (E_J + E0_J)  # (M_energy,)

    # 3.3 组合得到 wd 矩阵 (N_time x M_energy)
    # Wd = C * L * gamma * F(y)
    prefactor = (3 * CONSTANTS['RE']) / (CONSTANTS['QE'] * CONSTANTS['ME'])

    # 利用广播: (N,1) * (1,M) * (N,1) -> (N, M)
    wd_matrix_rad_s = prefactor * l_vals[:, None] * gamma_term[None, :] * F_y[:, None]
    wd_matrix = wd_matrix_rad_s * 3600 + CONSTANTS['W_COR_RAD_H']

    return {
        'time': time_vals,
        'L': l_vals,  # (N,)
        'E': e_centers,  # (M,)
        'Flux': flux_matrix,  # (N, M)
        'Wd': wd_matrix  # (N, M)
    }


# ==========================================
# 4. 绘图 (Direct Mesh Method)
# ==========================================
def plot_direct_mesh(data):
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), dpi=150)

    # 准备数据
    Time = data['time']
    L = data['L']
    E = data['E']
    Flux = data['Flux']
    Wd = data['Wd']

    # 颜色设置
    flux_valid = Flux[Flux > 0]
    vmin = np.nanpercentile(flux_valid, 5)
    vmax = np.nanpercentile(flux_valid, 99)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = 'jet'

    # --- 图 1: Time vs Energy (标准网格) ---
    ax1 = axes[0]
    # 使用 pcolormesh，X对应时间，Y对应能量
    # 注意：Flux.T 的形状是 (M, N)，对应 Y(M) 和 X(N)
    pcm1 = ax1.pcolormesh(Time, E, Flux.T, norm=norm, cmap=cmap, shading='auto')
    ax1.set_yscale('log')
    ax1.set_title('(a) Time vs Energy (Original Observations)')
    ax1.set_ylabel('Energy (keV)')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.colorbar(pcm1, ax=ax1, label='Flux')

    # --- 图 2: L vs Energy ---
    ax2 = axes[1]
    # X轴: L_sorted (N,), Y轴: E (M,)
    # Z轴: Flux_sorted.T (M, N)
    # 因为 X 和 Y 都是 1D 数组，pcolormesh 会自动构建网格
    pcm2 = ax2.pcolormesh(data["L"], E, Flux.T, norm=norm, cmap=cmap, shading='auto')
    ax2.set_yscale('log')
    ax2.set_title('(b) L vs Energy')
    ax2.set_xlim(1.2, 1.9)
    ax2.set_ylabel('Energy (keV)')
    ax2.set_xlabel('L-shell')
    plt.colorbar(pcm2, ax=ax2, label='Flux')

    # --- 图 3: L vs Omega_d (Curvilinear Grid / 非均匀网格) ---
    ax3 = axes[2]
    # 这里的 X 和 Y 都是二维矩阵！
    # X (L): 所有的能量通道共享同一个 L，所以要在 Energy 维度复制
    # L_grid shape: (M, N) - 对应 Flux.T 的形状
    L_grid = np.tile(L, (len(E), 1))

    # Y (Wd): 已经在上面计算好了，转置即可
    # Wd_grid shape: (M, N)
    Wd_grid = Wd.T

    # Z (Flux): (M, N)
    Z_data = Flux.T

    # 直接绘制！Matplotlib 会处理这些弯曲的四边形
    # shading='nearest' 或 'auto' 都可以，'flat' 需要网格比数据多一行
    # 对于这种不规则网格，'gouraud' (平滑) 有时会好看，但为了真实性用 'auto'
    pcm3 = ax3.pcolormesh(L_grid, Wd_grid, Z_data, norm=norm, cmap=cmap, shading='auto')

    ax3.set_title(r'(c) L vs $\omega_d$')
    ax3.set_ylabel(r'Drift Frequency $\omega_d$ (rad/h)')
    ax3.set_xlabel('L-shell')
    ax3.set_ylim(0, 4.0)  # 聚焦感兴趣区域

    # 叠加地球自转
    ax3.axhline(CONSTANTS['W_COR_RAD_H'], color='white', linestyle='--', label='Earth Corotation')
    ax3.legend(loc='upper right')

    plt.colorbar(pcm3, ax=ax3, label='Flux')

    plt.tight_layout()
    plt.show()


# ==========================================
# 5. 主程序
# ==========================================
if __name__ == "__main__":
    FILE_PATH = r'C:\Research\MSS1A_v1\MSS1A_20240524T045957_20240524T054530_FEDO_v1.1.txt'

    print("Loading...")
    try:
        df, e_cen, e_cols = load_fedo_data(FILE_PATH)
        data = calculate_physics_data(df, e_cen, e_cols)
        print("Plotting using direct mesh method...")
        plot_direct_mesh(data)
    except Exception as e:
        print(f"Error: {e}")