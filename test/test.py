import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.Preference.EnumType import ParaType
import h5py
# # 参数设置
# G = 6.67430e-11  # 引力常数
# M_Earth = 5.972e24  # 地球质量
# M_Sun = 1.989e30  # 太阳质量
# M_Moon = 7.342e22  # 月球质量
# C_d = 2.2  # 阻力系数
# A = 10  # 卫星截面积
# rho = 1e-12  # 大气密度
# P_srp = 4.56e-6  # 太阳光压
# m = 1000  # 卫星质量
#
# # 初始条件
# r0 = np.array([6778000, 0, 0])  # 初始位置
# v0 = np.array([0, 7600, 0])  # 初始速度
# dt = 10  # 时间步长
# t_end = 10000  # 仿真时间
# r_sun = 1
# r_moon = 2
# # 位置和速度的更新函数
# def acceleration(r, v):
#     # 计算总的加速度
#     F_g = -G * M_Earth * r / np.linalg.norm(r) ** 3  # 地球引力
#     F_sun = -G * M_Sun * (r - r_sun) / np.linalg.norm(r - r_sun) ** 3  # 太阳引力
#     F_moon = -G * M_Moon * (r - r_moon) / np.linalg.norm(r - r_moon) ** 3  # 月球引力
#     F_drag = -C_d * A * rho * v * np.linalg.norm(v) / 2  # 大气阻力
#     F_srp = P_srp * A * (r - r_sun) / np.linalg.norm(r - r_sun) ** 3  # 太阳光压
#
#     total_force = F_g + F_sun + F_moon + F_drag + F_srp
#     return total_force / m
#
#
# def rk4_step(r, v, dt):
#     k1_v = acceleration(r, v)
#     k1_r = v
#
#     k2_v = acceleration(r + 0.5 * dt * k1_r, v + 0.5 * dt * k1_v)
#     k2_r = v + 0.5 * dt * k1_v
#
#     k3_v = acceleration(r + 0.5 * dt * k2_r, v + 0.5 * dt * k2_v)
#     k3_r = v + 0.5 * dt * k2_v
#
#     k4_v = acceleration(r + dt * k3_r, v + dt * k3_v)
#     k4_r = v + dt * k3_v
#
#     r_new = r + dt * (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
#     v_new = v + dt * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
#
#     return r_new, v_new
#
#
# # 仿真
# r = r0
# v = v0
# t = 0
#
# # 存储轨迹
# trajectory = []
#
# while t < t_end:
#     trajectory.append(r)
#     r, v = rk4_step(r, v, dt)
#     t += dt
#
# # 转换为NumPy数组
# trajectory = np.array(trajectory)
#
# # 绘制轨迹
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Satellite Trajectory')
# ax1.set_xlabel('X (m)')
# ax1.set_ylabel('Y (m)')
# ax1.set_zlabel('Z (m)')
# ax1.legend()
# ax1.set_title('Satellite Orbit Simulation')
#
# # 选择一个特定的时间点进行力模型更新及轨道更新
# specific_time_index = len(trajectory) // 2  # 选择中间时刻
# specific_r = trajectory[specific_time_index]
# specific_v = (trajectory[specific_time_index + 1] - trajectory[specific_time_index - 1]) / (2 * dt)
#
# # 计算该时刻的力
# F_g = -G * M_Earth * specific_r / np.linalg.norm(specific_r) ** 3
# F_sun = -G * M_Sun * (specific_r - r_sun) / np.linalg.norm(specific_r - r_sun) ** 3
# F_moon = -G * M_Moon * (specific_r - r_moon) / np.linalg.norm(specific_r - r_moon) ** 3
# F_drag = -C_d * A * rho * specific_v * np.linalg.norm(specific_v) / 2
# F_srp = P_srp * A * (specific_r - r_sun) / np.linalg.norm(specific_r - r_sun) ** 3
#
# # 绘制力矢量
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Satellite Trajectory', alpha=0.5)
# ax2.quiver(specific_r[0], specific_r[1], specific_r[2], F_g[0], F_g[1], F_g[2], color='red', label='Earth Gravity')
# ax2.quiver(specific_r[0], specific_r[1], specific_r[2], F_sun[0], F_sun[1], F_sun[2], color='yellow',
#            label='Sun Gravity')
# ax2.quiver(specific_r[0], specific_r[1], specific_r[2], F_moon[0], F_moon[1], F_moon[2], color='gray',
#            label='Moon Gravity')
# ax2.quiver(specific_r[0], specific_r[1], specific_r[2], F_drag[0], F_drag[1], F_drag[2], color='blue',
#            label='Atmospheric Drag')
# ax2.quiver(specific_r[0], specific_r[1], specific_r[2], F_srp[0], F_srp[1], F_srp[2], color='orange',
#            label='Solar Radiation Pressure')
# ax2.set_xlabel('X (m)')
# ax2.set_ylabel('Y (m)')
# ax2.set_zlabel('Z (m)')
# ax2.legend()
# ax2.set_title('Force Vectors at a Specific Time')
#
# plt.tight_layout()
# plt.show()

# for key in [ParaType.TransitionMatrix, ParaType.Accelerometer, ParaType.StokesCoefficients]:
#     if key is ParaType.Accelerometer:
#         print(ParaType.Accelerometer)

filename = '../temp/CovarianceMatrix/2009-05-01-2009-05-31.hdf5'

h5 = h5py.File(filename, 'r')
cov = h5['cov'][:]
h5.close()

sigma = np.sqrt(np.diag(cov))

print(sigma)