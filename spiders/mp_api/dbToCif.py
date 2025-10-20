# -*- coding: utf-8 -*-
import json
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import colorsys
import numpy as np
from typing import Any, Dict, Iterable, Optional, Sequence, Union
import pyvista as pv
from pymatgen.core import Structure,Element
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.transformations.standard_transformations import OxidationStateDecorationTransformation
from pymatgen.analysis.bond_valence import BVAnalyzer
DATABASE_URL = "mysql+pymysql://root:123456@localhost:6116/docker-phalapi"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
def get_db():
    """获取数据库会话的工具函数"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


json_first_level_keys = ['old_material_id', 'builder_meta', 'nsites', 'elements', 'nelements', 'composition',
                         'composition_reduced', 'formula_pretty', 'formula_anonymous', 'chemsys', 'volume', 'density',
                         'density_atomic', 'symmetry', 'material_id', 'deprecated', 'deprecation_reasons',
                         'last_updated', 'origins', 'warnings', 'structure', 'property_name', 'task_ids',
                         'uncorrected_energy_per_atom', 'energy_per_atom', 'formation_energy_per_atom',
                         'energy_above_hull', 'is_stable', 'equilibrium_reaction_energy_per_atom', 'decomposes_to',
                         'xas', 'grain_boundaries', 'band_gap', 'cbm', 'vbm', 'efermi', 'is_gap_direct', 'is_metal',
                         'es_source_calc_id', 'bandstructure', 'dos', 'dos_energy_up', 'dos_energy_down', 'is_magnetic',
                         'ordering', 'total_magnetization', 'total_magnetization_normalized_vol',
                         'total_magnetization_normalized_formula_units', 'num_magnetic_sites',
                         'num_unique_magnetic_sites', 'types_of_magnetic_species', 'bulk_modulus', 'shear_modulus',
                         'universal_anisotropy', 'homogeneous_poisson', 'e_total', 'e_ionic', 'e_electronic', 'n',
                         'e_ij_max', 'weighted_surface_energy_EV_PER_ANG2', 'weighted_surface_energy',
                         'weighted_work_function', 'surface_anisotropy', 'shape_factor', 'has_reconstructed',
                         'possible_species', 'has_props', 'theoretical', 'database_IDs', 'fields_not_requested',
                         'set_time']
class MaterialPJ(Base):
    __tablename__ = "MaterialPJ"  # 表名
    # __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    for key in json_first_level_keys:
        if key in ["builder_meta", "bandstructure", "dos", "structure", "origins", "database_IDs", "symmetry",
                   "warnings", "task_ids", "xas", 'has_props']:
            locals()[key] = Column(Text, nullable=True)
        elif key in ["material_id", "old_material_id", "chemsys", "formula_pretty", "nsites", "nelements", "deprecated",
                     "is_stable"]:
            locals()[key] = Column(String(50), nullable=True)
        else:
            locals()[key] = Column(String(200), nullable=True)
    # 清理局部变量，避免冗余
    del key

def query_material_by_old_id(old_material_id):
    db = next(get_db())
    try:
        material = db.query(MaterialPJ).filter(MaterialPJ.old_material_id == old_material_id).all()
        return material
    finally:
        db.close()

# 1) 自定义元素颜色（十六进制）。未覆盖的元素将自动生成颜色
CUSTOM_ELEMENT_COLORS = {
    "Al": "#87CEEB",  # 天蓝
    "O":  "#FF6347",  # 番茄红
    "Fe":"#DAA520"}
    # "Si": "#40E0D0", # 可按需补充

# 2) 原子球半径模式： "vdw" | "fixed"
SPHERE_RADIUS_MODE = "vdw"    # 用范德华半径（更像球棒模型）
FIXED_RADIUS_ANGSTROM = 0.35  # 当选择 "fixed" 模式时生效
RADIUS_SCALE = 0.35           # 将 vdw 半径再缩放（防止太大）

# 3) 截图大小
SCREENSHOT_SIZE = (1200, 1000)
def _hex_to_rgb_uint8(hex_str: str, default=(150, 150, 150)) -> tuple:
    try:
        h = hex_str.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    except Exception:
        return default
def _auto_color_for_symbol(symbol: str) -> tuple:
    """
    未给定自定义颜色时，稳定地为元素生成一个颜色（按 symbol 做哈希）。
    返回 uint8 RGB 三元组。
    """
    # 用 symbol 生成 0-1 的 hue，固定饱和度/明度
    h = (hash(symbol) % 360) / 360.0
    s, v = 0.55, 0.95
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r*255), int(g*255), int(b*255))

def _color_for_symbol(symbol: str) -> tuple:
    if symbol in CUSTOM_ELEMENT_COLORS:
        return _hex_to_rgb_uint8(CUSTOM_ELEMENT_COLORS[symbol])
    return _auto_color_for_symbol(symbol)

def _load_structure(obj: Union[str, dict, Structure]) -> Structure:
    """
    接受三种类型：
    - JSON 字符串（数据库存的样子）
    - dict（已经 loads 后）
    - pymatgen.Structure（直接传结构）
    """
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, str):
        d = json.loads(obj)
        return Structure.from_dict(d)
    if isinstance(obj, dict):
        return Structure.from_dict(obj)
    raise TypeError(f"Unsupported structure type: {type(obj)}")

def _vdw_radius(symbol: str) -> float:
    """
    按元素取半径（vdW 为主，兜底 atomic）
    :param symbol:
    :return:
    """
    try:
        el = Element(symbol)
        return float(el.van_der_waals_radius or el.atomic_radius or 1.5)
    except Exception:
        return 1.5

def _sphere_radius_for_symbol(symbol: str, stick: bool = False) -> float:
    """
    支持球棒模式的原子半径
    :param symbol:
    :param stick:
    :return:
    """
    if SPHERE_RADIUS_MODE == "fixed":
        base = float(FIXED_RADIUS_ANGSTROM)
    else:
        # vdw 半径更像球棒模型的基线
        base = _vdw_radius(symbol) * float(RADIUS_SCALE)

    if stick:
        # 球棒模式下原子球更小（更像 VESTA）
        return base   # 可按需调 0.3~0.6
    return base



def build_bonds_mesh_crystalnn_vesta(structure, bond_radius=0.12,complement=False):
    """
    CrystalNN 成键，但在绘制时采用邻居的 periodic image 坐标，
    使连线落在最近镜像上，视觉效果更接近 VESTA。
    """
    coords = np.asarray(np.round(structure.cart_coords, 6), dtype=float)
    n = len(coords)
    if n <= 1:
        return None

    cnn = CrystalNN()
    pts = []
    lines = []
    if complement:
        for i in range(n):
            for info in cnn.get_nn_info(structure, i):
                j = info["site_index"]
                p1 = coords[i]
                try:
                    p2 = np.asarray(np.round(info["site"].coords, 6), dtype=float)
                except Exception:
                    p2 = coords[j]
                if j > i:
                    idx0 = len(pts)
                    pts.append(p1)
                    pts.append(p2)
                    lines.extend([2, idx0, idx0 + 1])
    else:
        for i in range(n):
            for info in cnn.get_nn_info(structure, i):
                j = info["site_index"]
                p1 = coords[i]
                try:
                    p2 =  np.asarray(np.round(info["site"].coords, 6), dtype=float)
                    # xdj=p2==p1
                    if np.any(np.all(coords == p2, axis=1)):
                        # print('1')
                        if j > i:
                            idx0 = len(pts)
                            pts.append(p1)
                            pts.append(p2)
                            lines.extend([2, idx0, idx0 + 1])
                    else:
                        continue
                except Exception:
                    print('p2不存在')
    if not pts:
        print('ok')
        return None
    print(pts)
    poly = pv.PolyData()
    poly.points = np.array(pts, dtype=float)
    poly.lines  = np.array(lines, dtype=int)
    return poly.tube(radius=bond_radius, n_sides=16)#侧面数量（即横截面的边数）

def _build_bonds_mesh(structure: Structure,
                      k_scale: float = 0.65,
                      bond_radius: float = 0.12) -> Optional[pv.PolyData]:
    """
    基于 vdW 半径阈值判定成键，并用 tube 生成圆柱棒。
    k_scale：阈值缩放，越大连的键越多（0.6~0.8 常用）
    bond_radius：棒的半径（Å）
    """
    coords = np.asarray(structure.cart_coords, dtype=float)
    n = len(coords)
    if n <= 1:
        return None

    # 预取每个原子的 vdW 半径
    syms = [str(s.specie) for s in structure.sites]
    r_vdw = np.array([_vdw_radius(sym) for sym in syms], dtype=float)
    # 距离矩阵（单位胞内）
    # 对于普通晶体单胞（几十/百来个原子）足够快；如需更大体系可换 neighbor_list
    dmat = structure.distance_matrix
    # 线段点与拓扑
    # lines vtk 格式：[2, i, j] * M
    pts = []
    lines = []

    # 简单 i<j 上三角扫描
    for i in range(n):
        for j in range(i + 1, n):
            # 阈值（vdW 和的比例）
            cutoff = k_scale * (r_vdw[i] + r_vdw[j])
            if dmat[i, j] <= cutoff:
                # 添加一条线段 i -> j
                idx0 = len(pts)
                pts.append(coords[i])
                pts.append(coords[j])
                lines.extend([2, idx0, idx0 + 1])

    if not pts:
        return None

    poly = pv.PolyData()
    poly.points = np.array(pts, dtype=float)
    poly.lines = np.array(lines, dtype=int)
    # tube 成圆柱棒
    tube = poly.tube(radius=bond_radius, n_sides=16)
    return tube

def _build_cell_edges(structure: Structure) -> pv.PolyData:
    """
    根据晶格基矢，构建平行六面体的 8 个顶点 + 12 条边。
    """
    a_vec, b_vec, c_vec = structure.lattice.matrix  # 3x3
    O= np.array([0, 0, 0])
    A= a_vec
    B= b_vec
    C= c_vec
    AB= a_vec + b_vec
    AC= a_vec + c_vec
    BC= b_vec + c_vec
    ABC = a_vec + b_vec + c_vec

    corners = np.vstack([O, A, B, AB, C, AC, BC, ABC])
    edges_pairs = [
        (0,1), (0,2), (1,3), (2,3),   # 底面
        (4,5), (4,6), (5,7), (6,7),   # 顶面
        (0,4), (1,5), (2,6), (3,7)    # 立柱
    ]
    lines = []
    for i, j in edges_pairs:
        lines.extend([2, i, j])
    lines = np.array(lines)

    cell_poly = pv.PolyData()
    cell_poly.points = corners
    cell_poly.lines = lines
    return cell_poly

def _add_atoms_by_groups(plotter: pv.Plotter, structure: Structure, stick: bool = False):
    cart = np.asarray(structure.cart_coords, dtype=float)
    syms = [str(site.specie) for site in structure.sites]

    from collections import defaultdict
    groups = defaultdict(list)
    for idx, sym in enumerate(syms):
        groups[sym].append(idx)

    legend_entries = []
    for sym, idxs in groups.items():
        pts = cart[np.asarray(idxs, dtype=int)]
        color = _color_for_symbol(sym)
        radius = _sphere_radius_for_symbol(sym, stick=stick)

        pdata = pv.PolyData(pts)
        sphere = pv.Sphere(radius=radius, theta_resolution=24, phi_resolution=24)
        mesh = pdata.glyph(geom=sphere, orient=False, scale=False)

        plotter.add_mesh(
            mesh,
            color=np.array(color) / 255.0,
            smooth_shading=True,
            name=f"atoms_{sym}",
        )
        
        # 添加图例条目
        legend_entries.append([sym, list(np.array(color) / 255.0)])
    
    return legend_entries

def Oxidation_edit(structure, Oxidation_dict=None):
    """
    为 pymatgen 结构对象分配氧化态。
    如果提供 Oxidation_dict（例如 {"Fe": 3, "O": -2}），使用该字典；
    否则自动推断氧化态。
    """
    try:
        if Oxidation_dict:  # 用户指定氧化态
            osd = OxidationStateDecorationTransformation(Oxidation_dict)
            structure = osd.apply_transformation(structure)
            print("已根据输入氧化态字典装饰结构")
        else:
            # 自动分配氧化态
            analyzer = BVAnalyzer()
            structure = analyzer.get_oxi_state_decorated_structure(structure)
            print("已自动推断氧化态（Bond Valence 方法）")
    except Exception as e:
        print(f"氧化态分配失败: {e}")
    finally:
        return structure
def render_structure(
    structure_in: Union[str, dict, Structure],
    out_path: Optional[str] = None,
    interactive: bool = False,
    show_axes: bool = True,
    background: str = "white",
    stick: bool = False,
    bond_method: str = "radius",   # "radius" | "crystalnn"
    bond_k: float = 0.65,          # 仅对 bond_method="radius" 生效
    bond_radius: float = 0.12,     # “棒”的圆柱半径
    primitive_cell: bool = False,
    complement=False
) -> Optional[str]:
    """
    渲染结构；支持球棒模型，并在 'radius' 与 'crystalnn' 两种成键方法之间切换。
    依赖的外部函数/变量：
      - _load_structure()
      - _add_atoms_by_groups(plotter, structure, stick)
      - _build_cell_edges(structure)
      - _build_bonds_mesh(structure, k_scale, bond_radius)  # 你已有的半径阈值法
      - SCREENSHOT_SIZE, pv.global_theme.background
    """
    structure = _load_structure(structure_in)

    if primitive_cell:
        sga = SpacegroupAnalyzer(structure)
        structure = sga.get_conventional_standard_structure()  # 转换为惯用晶胞

    # 检查结构中是否包含氧元素，如果包含则自动分配氧化态
    if 'O' in [str(e) for e in structure.composition.elements]:
        print("检测到结构中包含氧元素，将自动分配氧化态")
        structure=Oxidation_edit( structure)

    pv.global_theme.background = background
    plotter = pv.Plotter(off_screen=not interactive, window_size=SCREENSHOT_SIZE)

    # 1) 原子（球）
    legend_entries = _add_atoms_by_groups(plotter, structure, stick=stick)

    # 2) 晶胞线框
    cell_poly = _build_cell_edges(structure)
    plotter.add_mesh(cell_poly, color="black", line_width=2, name="Lattice")

    # 3) 添加图例
    if legend_entries:
        plotter.add_legend(legend_entries, loc="upper right")

    # 3) 球棒模式下添加“棒”
    if stick:
        bonds_mesh = None
        if bond_method.lower() == "crystalnn":
            bonds_mesh = build_bonds_mesh_crystalnn_vesta(structure, bond_radius=bond_radius,complement=complement)

        elif bond_method.lower() == "radius":
            # 你之前实现的半径阈值函数：_build_bonds_mesh(structure, k_scale, bond_radius)
            bonds_mesh = _build_bonds_mesh(structure, k_scale=bond_k, bond_radius=bond_radius)
        else:
            raise ValueError("bond_method 必须为 'radius' 或 'crystalnn'。")

        if bonds_mesh is not None:
            plotter.add_mesh(bonds_mesh, color="gray", smooth_shading=True, name=f"Bonds[{bond_method}]")
            # if bond_method.lower() == "crystalnn":
            #     _add_crystalnn_ghost_atoms(plotter, structure, opacity=0.45, scale=0.8, stick=stick)

    # 4) 视图、坐标轴
    if show_axes:
        plotter.add_axes(xlabel="a", ylabel="b", zlabel="c")
    plotter.show_grid(color="lightgray")
    plotter.camera.zoom(1.25)

    # 5) 交互 / 截图
    if interactive:
        plotter.show()
        return None
    else:
        if out_path is None:
            raise ValueError("离屏模式下必须提供 out_path 保存截图。")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(out_path)
        plotter.close()
        return out_path



def _add_crystalnn_ghost_atoms(plotter: pv.Plotter,
                               structure: Structure,
                               opacity: float = 0.45,
                               scale: float = 0.8,
                               stick: bool = True) -> None:
    from collections import defaultdict
    cnn = CrystalNN()
    lattice = structure.lattice
    coords = np.asarray(structure.cart_coords, dtype=float)
    eps = 1e-6

    ghost_by_sym = defaultdict(list)
    for i in range(len(structure)):
        for info in cnn.get_nn_info(structure, i):
            site2 = info.get("site")
            if site2 is None:
                continue
            p2 = np.asarray(site2.coords, dtype=float)
            f2 = np.asarray(lattice.get_fractional_coords(p2), dtype=float)
            if (f2 < -eps).any() or (f2 >= 1.0 + eps).any():
                if (np.linalg.norm(coords - p2, axis=1) > 1e-3).all():
                    ghost_by_sym[str(site2.specie)].append(p2)

    if not ghost_by_sym:
        return

    for sym, pts in ghost_by_sym.items():
        pdata = pv.PolyData(np.asarray(pts, dtype=float))
        r = _sphere_radius_for_symbol(sym, stick=stick) * float(scale)
        sphere = pv.Sphere(radius=r, theta_resolution=20, phi_resolution=20)
        mesh = pdata.glyph(geom=sphere, orient=False, scale=False)
        color = np.array(_color_for_symbol(sym)) / 255.0
        plotter.add_mesh(mesh, color=color, opacity=float(opacity), smooth_shading=True, name=f"ghost_{sym}")


def batch_render(
    records: Sequence[Dict[str, Any]],
    out_dir: str = "./renders",
    stick: bool = False,                 # ← 新增
    interactive: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for rec in records:
        sid = rec.old_material_id
        sdata = rec.structure
        out_path = os.path.join(out_dir, f"{sid}.png")
        try:
            render_structure(
                sdata,
                out_path=None if interactive else out_path,
                interactive=interactive,
                stick=stick,             # ← 传入
            )
            if not interactive:
                print(f"[OK] {sid} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")


if __name__ == "__main__":
    # 'mp-1180433'
    # 'mp-19770'
    reconds=query_material_by_old_id('mp-1180433')
    render_structure(
        reconds[0].structure,
        out_path="renders/b.png",
        interactive=True,
        stick=True,
        bond_method="crystalnn",
        bond_radius=0.10,
        bond_k=0.59,
        primitive_cell=False,
        complement=False)
