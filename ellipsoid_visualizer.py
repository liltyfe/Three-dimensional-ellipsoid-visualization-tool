import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib import font_manager
import os

font_prop = None
font_paths = [
    r'C:\Windows\Fonts\simhei.ttf',
    r'C:\Windows\Fonts\msyh.ttc',
    r'C:\Windows\Fonts\simsun.ttc',
    r'C:\Windows\Fonts\simkai.ttf',
]

for fp in font_paths:
    if os.path.exists(fp):
        try:
            font_prop = font_manager.FontProperties(fname=fp)
            print(f"找到字体: {fp}")
            break
        except:
            continue

if font_prop is None:
    print("未找到中文字体，使用默认")

matplotlib.rcParams['axes.unicode_minus'] = False

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def create_ellipsoid(A, center=[0, 0, 0], n_points=50):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    a = 1.0 / np.sqrt(eigenvalues[0])
    b = 1.0 / np.sqrt(eigenvalues[1])
    c = 1.0 / np.sqrt(eigenvalues[2])
    
    u = np.linspace(0, 2 * np.pi, n_points)
    v = np.linspace(0, np.pi, n_points)
    
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))
    
    for i in range(len(x)):
        for j in range(len(x)):
            point = np.array([x[i, j], y[i, j], z[i, j]])
            rotated = eigenvectors @ point
            x[i, j] = rotated[0] + center[0]
            y[i, j] = rotated[1] + center[1]
            z[i, j] = rotated[2] + center[2]
    
    return x, y, z, eigenvalues, eigenvectors, (a, b, c)

def matrix_to_params(A):
    return [A[0, 0], A[1, 1], A[2, 2], A[0, 1], A[0, 2], A[1, 2]]

def params_to_matrix(a11, a22, a33, a12, a13, a23):
    return np.array([
        [a11, a12, a13],
        [a12, a22, a23],
        [a13, a23, a33]
    ])

class EllipsoidVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        if font_prop:
            self.fig.suptitle('二次型矩阵与三维椭球可视化', fontproperties=font_prop, fontsize=14, fontweight='bold')
        else:
            self.fig.suptitle('Quadratic Form Matrix and 3D Ellipsoid Visualization', fontsize=14, fontweight='bold')
        
        self.ax = self.fig.add_subplot(121, projection='3d')
        
        self.ax_info = self.fig.add_subplot(122)
        self.ax_info.axis('off')
        
        self.A = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        self.setup_sliders()
        self.setup_buttons()
        self.update_plot(None)
        
    def setup_sliders(self):
        slider_color = 'lightgoldenrodyellow'
        
        slider_axes = []
        labels = ['A11', 'A22', 'A33', 'A12', 'A13', 'A23']
        initials = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        
        for i in range(6):
            ax_slider = self.fig.add_axes([0.15, 0.02 + i * 0.035, 0.25, 0.02], 
                                          facecolor=slider_color)
            slider_axes.append(ax_slider)
        
        self.sliders = []
        for i, (ax, label, init) in enumerate(zip(slider_axes, labels, initials)):
            slider = Slider(ax, label, -2.0, 3.0, valinit=init, valstep=0.1)
            if font_prop:
                slider.label.set_fontproperties(font_prop)
            slider.on_changed(self.update_plot)
            self.sliders.append(slider)
    
    def setup_buttons(self):
        ax_reset = self.fig.add_axes([0.45, 0.02, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset' if font_prop is None else '重置', color='lightblue', hovercolor='skyblue')
        if font_prop:
            self.btn_reset.label.set_fontproperties(font_prop)
        self.btn_reset.on_clicked(self.reset)
        
        ax_sphere = self.fig.add_axes([0.45, 0.07, 0.08, 0.04])
        self.btn_sphere = Button(ax_sphere, 'Sphere' if font_prop is None else '球体', color='lightgreen', hovercolor='green')
        if font_prop:
            self.btn_sphere.label.set_fontproperties(font_prop)
        self.btn_sphere.on_clicked(self.set_sphere)
        
        ax_ellipsoid = self.fig.add_axes([0.45, 0.12, 0.08, 0.04])
        self.btn_ellipsoid = Button(ax_ellipsoid, 'Ellipsoid' if font_prop is None else '椭球', color='lightyellow', hovercolor='yellow')
        if font_prop:
            self.btn_ellipsoid.label.set_fontproperties(font_prop)
        self.btn_ellipsoid.on_clicked(self.set_ellipsoid)
        
        ax_rotated = self.fig.add_axes([0.45, 0.17, 0.08, 0.04])
        self.btn_rotated = Button(ax_rotated, 'Rotated' if font_prop is None else '旋转椭球', color='lightcoral', hovercolor='coral')
        if font_prop:
            self.btn_rotated.label.set_fontproperties(font_prop)
        self.btn_rotated.on_clicked(self.set_rotated)
    
    def update_plot(self, val):
        self.ax.clear()
        
        params = [slider.val for slider in self.sliders]
        self.A = params_to_matrix(*params)
        
        try:
            eigenvalues = np.linalg.eigvalsh(self.A)
            is_positive_definite = np.all(eigenvalues > 0)
        except:
            is_positive_definite = False
        
        if is_positive_definite:
            x, y, z, eigenvalues, eigenvectors, axes_lengths = create_ellipsoid(self.A)
            
            self.ax.plot_surface(x, y, z, alpha=0.6, cmap='coolwarm', 
                                edgecolor='none', antialiased=True)
            
            colors = ['red', 'green', 'blue']
            for i in range(3):
                scale = axes_lengths[i] * 0.8
                direction = eigenvectors[:, i] * scale
                arrow = Arrow3D([0, direction[0]], [0, direction[1]], [0, direction[2]],
                               mutation_scale=15, lw=2, arrowstyle='-|>', color=colors[i])
                self.ax.add_artist(arrow)
            
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            if font_prop:
                self.ax.set_title('椭球形状 (x^T A x = 1)', fontproperties=font_prop)
            else:
                self.ax.set_title('Ellipsoid Shape (x^T A x = 1)')
            
            max_axis = max(axes_lengths) * 1.5
            self.ax.set_xlim([-max_axis, max_axis])
            self.ax.set_ylim([-max_axis, max_axis])
            self.ax.set_zlim([-max_axis, max_axis])
            
            self.update_info_panel(eigenvalues, eigenvectors, axes_lengths, is_positive_definite=True)
        else:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            if font_prop:
                self.ax.set_title('矩阵非正定，无法形成椭球', fontproperties=font_prop)
            else:
                self.ax.set_title('Matrix not positive definite')
            self.ax.text(0, 0, 0, 'No valid ellipsoid', fontsize=12, ha='center')
            self.update_info_panel(None, None, None, is_positive_definite=False)
        
        self.fig.canvas.draw_idle()
    
    def update_info_panel(self, eigenvalues, eigenvectors, axes_lengths, is_positive_definite):
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = "[Quadratic Matrix A]\n\n" if font_prop is None else "[二次型矩阵 A]\n\n"
        info_text += f"┌                    ┐\n"
        info_text += f"│ {self.A[0,0]:6.2f}  {self.A[0,1]:6.2f}  {self.A[0,2]:6.2f} │\n"
        info_text += f"│ {self.A[1,0]:6.2f}  {self.A[1,1]:6.2f}  {self.A[1,2]:6.2f} │\n"
        info_text += f"│ {self.A[2,0]:6.2f}  {self.A[2,1]:6.2f}  {self.A[2,2]:6.2f} │\n"
        info_text += f"└                    ┘\n\n"
        
        if is_positive_definite:
            if font_prop:
                info_text += "[特征值与主轴]\n\n"
            else:
                info_text += "[Eigenvalues & Axes]\n\n"
            for i in range(3):
                info_text += f"Eigenvalue λ{i+1} = {eigenvalues[i]:.3f}\n"
                info_text += f"  Axis Length = {axes_lengths[i]:.3f}\n"
                info_text += f"  Eigenvector = [{eigenvectors[0,i]:.2f}, {eigenvectors[1,i]:.2f}, {eigenvectors[2,i]:.2f}]\n\n"
            
            if font_prop:
                info_text += "[几何意义]\n\n"
                info_text += "• 特征值越大 → 该方向主轴越短\n"
                info_text += "• 特征值越小 → 该方向主轴越长\n"
                info_text += "• 特征向量 → 主轴的方向\n"
            else:
                info_text += "[Geometry]\n\n"
                info_text += "• Larger eigenvalue → shorter axis\n"
                info_text += "• Smaller eigenvalue → longer axis\n"
                info_text += "• Eigenvector = axis direction\n"
            
            self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                            fontsize=10, verticalalignment='top',
                            fontproperties=font_prop if font_prop else None,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            if font_prop:
                info_text += "[警告] 矩阵非正定！\n\n"
            else:
                info_text += "[Warning] Matrix not positive definite!\n\n"
            try:
                eigs = np.linalg.eigvalsh(self.A)
                for i, e in enumerate(eigs):
                    status = "OK" if e > 0 else "BAD"
                    info_text += f"  λ{i+1} = {e:.3f} {status}\n"
            except:
                info_text += "  Failed to compute\n"
            
            self.ax_info.text(0.05, 0.95, info_text, transform=self.ax_info.transAxes,
                            fontsize=10, verticalalignment='top',
                            fontproperties=font_prop if font_prop else None,
                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    def reset(self, event):
        for slider in self.sliders:
            slider.reset()
    
    def set_sphere(self, event):
        values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
        for slider, val in zip(self.sliders, values):
            slider.set_val(val)
    
    def set_ellipsoid(self, event):
        values = [0.25, 1.0, 4.0, 0.0, 0.0, 0.0]
        for slider, val in zip(self.sliders, values):
            slider.set_val(val)
    
    def set_rotated(self, event):
        values = [1.0, 1.0, 4.0, 0.5, 0.3, 0.0]
        for slider, val in zip(self.sliders, values):
            slider.set_val(val)
    
    def show(self):
        plt.show()

def main():
    print("=" * 60)
    if font_prop:
        print("二次型矩阵与三维椭球可视化工具")
    else:
        print("Quadratic Form Matrix and 3D Ellipsoid Visualization")
    print("=" * 60)
    print("\nEllipsoid equation: x^T A x = 1")
    if font_prop:
        print("\n操作说明:")
        print("  • 拖动滑块调整矩阵A的6个独立元素")
        print("  • A11, A22, A33: 对角元素，控制各轴缩放")
        print("  • A12, A13, A23: 非对角元素，控制椭球旋转")
    else:
        print("\nInstructions:")
        print("  • Drag sliders to adjust 6 independent elements of matrix A")
        print("  • A11, A22, A33: diagonal elements, control scaling")
        print("  • A12, A13, A23: off-diagonal, control rotation")
    print("=" * 60)
    
    viz = EllipsoidVisualizer()
    viz.show()

if __name__ == "__main__":
    main()
