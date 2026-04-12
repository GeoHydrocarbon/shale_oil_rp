from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd


def load_rockphypy_modules(base_dir: Path, config: dict):
    # 只加载本次建模真正需要的子模块，避免触发不相关依赖。
    repo_path = (base_dir / config["paths"]["rockphypy_repo"]).resolve()
    package_dir = repo_path / "rockphypy"
    if not package_dir.exists():
        raise FileNotFoundError(f"rockphypy package not found: {package_dir}")

    package_name = "rockphypy"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(package_dir)]
        sys.modules[package_name] = package

    module_paths = {
        "utils": package_dir / "utils.py",
        "BW": package_dir / "BW.py",
        "Fluid": package_dir / "Fluid.py",
        "EM": package_dir / "EM.py",
    }
    loaded = {}
    for short_name, module_path in module_paths.items():
        full_name = f"{package_name}.{short_name}"
        module = sys.modules.get(full_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(full_name, module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load module: {full_name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        setattr(package, short_name, module)
        loaded[short_name] = module

    return loaded["BW"].BW, loaded["EM"].EM, loaded["Fluid"].Fluid, loaded["utils"].utils


def select_average(mode: str, voigt: np.ndarray, reuss: np.ndarray, hill: np.ndarray) -> np.ndarray:
    mode_lower = mode.lower()
    if mode_lower == "voigt":
        return voigt
    if mode_lower == "reuss":
        return reuss
    if mode_lower == "hill":
        return hill
    raise ValueError(f"Unsupported matrix_mixing mode: {mode}")


def compute_matrix_moduli(frame: pd.DataFrame, config: dict, EM) -> tuple[np.ndarray, np.ndarray]:
    # 根据无机矿物与干酪根体积分数计算等效基质模量。
    mineral_names = list(config["columns"]["minerals"].keys())
    component_names = mineral_names + ["KEROGEN"]
    volume_columns = [f"{name}_vol" for name in component_names]
    valid = frame["solid_valid"].to_numpy(dtype=bool)

    k_matrix = np.full(len(frame), np.nan)
    g_matrix = np.full(len(frame), np.nan)

    if valid.any():
        volumes = frame.loc[valid, volume_columns].to_numpy(dtype=float)
        k_values = np.array(
            [config["minerals"][name]["K"] for name in mineral_names] + [config["kerogen"]["K"]],
            dtype=float,
        )
        g_values = np.array(
            [config["minerals"][name]["G"] for name in mineral_names] + [config["kerogen"]["G"]],
            dtype=float,
        )

        k_voigt, k_reuss, k_hill = EM.VRH(volumes, k_values)
        g_voigt, g_reuss, g_hill = EM.VRH(volumes, g_values)

        k_matrix[valid] = select_average(config["model"]["matrix_mixing"], k_voigt, k_reuss, k_hill)
        g_matrix[valid] = select_average(config["model"]["matrix_mixing"], g_voigt, g_reuss, g_hill)

    return k_matrix, g_matrix


def compute_fluid_properties(frame: pd.DataFrame, config: dict, BW) -> tuple[np.ndarray, np.ndarray]:
    # 根据温压、矿化度和含水饱和度计算油水混合流体性质。
    fluid_cfg = config["fluid"]
    system = fluid_cfg["system"].lower()
    if system != "oil_water":
        raise ValueError(f"Unsupported fluid system: {fluid_cfg['system']}")

    temperature = float(fluid_cfg["temperature_c"])
    pressure = float(fluid_cfg["pressure_mpa"])
    salinity = float(fluid_cfg["brine_salinity_ppm"]) / 1e6
    oil_density = float(fluid_cfg["oil_density_gcc"])
    gas_gravity = float(fluid_cfg["gas_gravity"])
    gas_oil_ratio = float(fluid_cfg["gas_oil_ratio_l_l"])

    rho_brine, k_brine = BW.rho_K_brine(temperature, pressure, salinity)
    if gas_oil_ratio > 0.0:
        rho_oil, k_oil = BW.rho_K_go(pressure, temperature, oil_density, gas_gravity, gas_oil_ratio)
    else:
        rho_oil, k_oil = BW.rho_K_oil(pressure, temperature, oil_density)

    sw = frame["sw"].to_numpy(dtype=float)
    so = 1.0 - sw
    rho_fluid = sw * rho_brine + so * rho_oil

    mixing_mode = config["model"]["fluid_mixing"].lower()
    if mixing_mode != "reuss":
        raise ValueError(f"Unsupported fluid_mixing mode: {config['model']['fluid_mixing']}")
    k_fluid = 1.0 / (sw / k_brine + so / k_oil)

    return rho_fluid, k_fluid


def run_dem_step(k_host: float, g_host: float, alpha: float, phi_target: float, EM) -> tuple[float, float]:
    if phi_target <= 0.0:
        return k_host, g_host
    k_path, g_path, t = EM.Berryman_DEM(k_host, g_host, 0.0, 0.0, alpha, phi_target)
    return float(np.interp(phi_target, t, k_path)), float(np.interp(phi_target, t, g_path))


def compute_dry_frame(
    k_matrix: np.ndarray,
    g_matrix: np.ndarray,
    phi: np.ndarray,
    valid: np.ndarray,
    config: dict,
    EM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # 根据硬孔和软孔两类孔隙的纵横比，顺序估算干岩骨架模量。
    dry_model = config["model"]["dry_frame_model"].lower()
    pore_cfg = config.get("pore_system", {})

    k_dry = np.full(len(phi), np.nan)
    g_dry = np.full(len(phi), np.nan)
    phi_hard = np.full(len(phi), np.nan)
    phi_soft = np.full(len(phi), np.nan)

    if dry_model == "cripor":
        phic = float(config["model"]["critical_porosity"])
        active = valid & (phi >= 0.0) & (phi < phic)
        phi_hard[active] = phi[active]
        phi_soft[active] = 0.0
        if active.any():
            k_dry[active], g_dry[active] = EM.cripor(k_matrix[active], g_matrix[active], phi[active], phic)
        return k_dry, g_dry, phi_hard, phi_soft

    if dry_model != "dem":
        raise ValueError(f"Unsupported dry_frame_model: {config['model']['dry_frame_model']}")

    active = valid & (phi >= 0.0) & (phi < 1.0)
    pore_model = pore_cfg.get("model", "single_pore").lower()
    soft_fraction = float(pore_cfg.get("soft_pore_fraction", 0.0))
    soft_fraction = min(max(soft_fraction, 0.0), 1.0)
    alpha_hard = float(pore_cfg.get("alpha_hard", config["model"].get("pore_aspect_ratio", 0.08)))
    alpha_soft = float(pore_cfg.get("alpha_soft", config["model"].get("pore_aspect_ratio", 0.08)))
    dem_order = pore_cfg.get("dem_order", "hard_then_soft").lower()
    alpha_single = float(config["model"].get("pore_aspect_ratio", 0.08))

    for index in np.where(active)[0]:
        if pore_model == "dual_pore":
            phi_soft[index] = phi[index] * soft_fraction
            phi_hard[index] = phi[index] - phi_soft[index]

            if dem_order == "hard_then_soft":
                phi_hard_stage = phi_hard[index] / max(1.0 - phi_soft[index], 1e-8)
                k_step, g_step = run_dem_step(k_matrix[index], g_matrix[index], alpha_hard, phi_hard_stage, EM)
                k_dry[index], g_dry[index] = run_dem_step(k_step, g_step, alpha_soft, phi_soft[index], EM)
            elif dem_order == "soft_then_hard":
                phi_soft_stage = phi_soft[index] / max(1.0 - phi_hard[index], 1e-8)
                k_step, g_step = run_dem_step(k_matrix[index], g_matrix[index], alpha_soft, phi_soft_stage, EM)
                k_dry[index], g_dry[index] = run_dem_step(k_step, g_step, alpha_hard, phi_hard[index], EM)
            else:
                raise ValueError(f"Unsupported pore_system.dem_order: {pore_cfg['dem_order']}")
        else:
            phi_hard[index] = phi[index]
            phi_soft[index] = 0.0
            k_dry[index], g_dry[index] = run_dem_step(k_matrix[index], g_matrix[index], alpha_single, phi[index], EM)

    return k_dry, g_dry, phi_hard, phi_soft


def run_model(frame: pd.DataFrame, config: dict, base_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    # 第一步：加载 rockphypy 中本次建模真正需要的模块。
    BW, EM, Fluid, utils = load_rockphypy_modules(base_dir, config)

    result = frame.copy()
    valid = result["model_valid"].to_numpy(dtype=bool)
    phi = result["phi"].to_numpy(dtype=float)

    # 第二步：由矿物和干酪根体积分数计算无孔基质的等效模量和基质密度。
    k_matrix, g_matrix = compute_matrix_moduli(result, config, EM)
    result["K_matrix_GPa"] = k_matrix
    result["G_matrix_GPa"] = g_matrix
    result["rho_matrix_gcc"] = result["matrix_grain_density"]

    # 第三步：根据地层流体参数和饱和度计算孔隙流体性质。
    rho_fluid, k_fluid = compute_fluid_properties(result, config, BW)
    result["rho_fluid_gcc"] = rho_fluid
    result["K_fluid_GPa"] = k_fluid

    # 第四步：将总孔隙拆为硬孔和软孔，并用双孔隙 DEM 估算干岩骨架模量。
    k_dry, g_dry, phi_hard, phi_soft = compute_dry_frame(k_matrix, g_matrix, phi, valid, config, EM)
    result["phi_hard"] = phi_hard
    result["phi_soft"] = phi_soft
    result["K_dry_GPa"] = k_dry
    result["G_dry_GPa"] = g_dry

    # 第五步：用 Gassmann 低频流体替换得到饱和岩石模量。
    sat_valid = valid & np.isfinite(k_dry) & np.isfinite(g_dry) & np.isfinite(k_fluid)
    k_sat = np.full(len(result), np.nan)
    g_sat = np.full(len(result), np.nan)
    rho_model = np.full(len(result), np.nan)
    vp_model = np.full(len(result), np.nan)
    vs_model = np.full(len(result), np.nan)

    if sat_valid.any():
        k_sat[sat_valid], g_sat[sat_valid] = Fluid.Gassmann(
            k_dry[sat_valid],
            g_dry[sat_valid],
            k_matrix[sat_valid],
            k_fluid[sat_valid],
            phi[sat_valid],
        )
        rho_model[sat_valid] = (
            (1.0 - phi[sat_valid]) * result.loc[sat_valid, "rho_matrix_gcc"].to_numpy(dtype=float)
            + phi[sat_valid] * rho_fluid[sat_valid]
        )

        # 第六步：由饱和模量和密度计算模型预测的纵横波速度。
        vp_model[sat_valid], vs_model[sat_valid] = utils.V(k_sat[sat_valid], g_sat[sat_valid], rho_model[sat_valid])

    result["K_sat_GPa"] = k_sat
    result["G_sat_GPa"] = g_sat
    result["rho_model_gcc"] = rho_model
    result["vp_model_m_s"] = vp_model
    result["vs_model_m_s"] = vs_model

    # 第七步：与测井实测值做差，形成后续标定所需的误差列。
    result["vp_misfit_m_s"] = result["vp_model_m_s"] - result["vp_log_m_s"]
    result["vs_misfit_m_s"] = result["vs_model_m_s"] - result["vs_log_m_s"]
    result["rho_misfit_gcc"] = result["rho_model_gcc"] - result["rho_log_gcc"]
    result["compare_valid"] = sat_valid & result["sonic_valid"].to_numpy(dtype=bool) & result["density_valid"].to_numpy(dtype=bool)

    summary = {
        "rows": int(len(result)),
        "modeled_rows": int(np.isfinite(result["vp_model_m_s"]).sum()),
        "compare_rows": int(result["compare_valid"].sum()),
    }
    return result, summary
