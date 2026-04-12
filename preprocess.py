from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def parse_sheet_name(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    return int(value) if str(value).isdigit() else value


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def to_fraction(series: pd.Series, unit: str) -> pd.Series:
    data = to_numeric(series)
    unit_lower = unit.lower()
    if unit_lower in {"fraction", "frac"}:
        return data
    if unit_lower in {"percent", "%"}:
        return data / 100.0
    raise ValueError(f"Unsupported fraction unit: {unit}")


def to_density_gcc(series: pd.Series, unit: str) -> pd.Series:
    data = to_numeric(series)
    unit_lower = unit.lower()
    if unit_lower in {"g/cm3", "g/cc", "gcc"}:
        return data
    if unit_lower in {"kg/m3", "kg/m^3"}:
        return data / 1000.0
    raise ValueError(f"Unsupported density unit: {unit}")


def sonic_to_velocity(series: pd.Series, unit: str) -> pd.Series:
    dt = to_numeric(series)
    dt = dt.where(dt > 0.0)
    unit_lower = unit.lower()
    if unit_lower in {"us/ft", "us_per_ft"}:
        return 304800.0 / dt
    if unit_lower in {"us/m", "us_per_m"}:
        return 1_000_000.0 / dt
    raise ValueError(f"Unsupported sonic unit: {unit}")


def clip_and_normalize(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    cleaned = data.copy().fillna(0.0)
    negative_count = (cleaned < 0.0).sum()
    cleaned = cleaned.clip(lower=0.0)
    row_sum_before_norm = cleaned.sum(axis=1)
    normalized = cleaned.div(row_sum_before_norm.where(row_sum_before_norm > 0.0), axis=0).fillna(0.0)
    return normalized, row_sum_before_norm, negative_count


def weight_to_volume_fraction(
    weight_fraction: pd.DataFrame,
    density_map: dict[str, float],
) -> tuple[pd.DataFrame, pd.Series]:
    densities = pd.Series(density_map)
    raw_volume = weight_fraction.div(densities, axis=1)
    raw_volume_sum = raw_volume.sum(axis=1)
    volume_fraction = raw_volume.div(raw_volume_sum.where(raw_volume_sum > 0.0), axis=0).fillna(0.0)
    return volume_fraction, raw_volume_sum


def compute_log_moduli(vp: pd.Series, vs: pd.Series, rho: pd.Series) -> tuple[pd.Series, pd.Series]:
    k = pd.Series(np.nan, index=vp.index, dtype=float)
    g = pd.Series(np.nan, index=vp.index, dtype=float)
    valid = vp.notna() & vs.notna() & rho.notna() & (vp > 0.0) & (vs > 0.0) & (rho > 0.0)
    k.loc[valid] = rho.loc[valid] * 1000.0 * (vp.loc[valid] ** 2 - 4.0 * vs.loc[valid] ** 2 / 3.0) / 1e9
    g.loc[valid] = rho.loc[valid] * 1000.0 * vs.loc[valid] ** 2 / 1e9
    return k, g


def preprocess_dataframe(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, int]]:
    columns = config["columns"]
    units = config["units"]
    preprocess_cfg = config["preprocess"]
    minerals_cfg = config["minerals"]

    result = frame.copy()
    # 第一步：统一原始测井数据单位，生成后续建模直接可用的标准列。
    result["depth_m"] = to_numeric(result[columns["depth"]])
    result["vp_log_m_s"] = sonic_to_velocity(result[columns["dtp"]], units["dtp"])
    result["vs_log_m_s"] = sonic_to_velocity(result[columns["dts"]], units["dts"])
    result["rho_log_gcc"] = to_density_gcc(result[columns["density"]], units["density"])
    result["phi"] = to_fraction(result[columns["porosity"]], units["porosity"])
    result["sw"] = to_fraction(result[columns["water_saturation"]], units["water_saturation"])
    result["so"] = 1.0 - result["sw"]

    # 第二步：清洗 TOC 等辅助曲线，先保留但暂不直接并入基质模型。
    toc_column = columns.get("toc")
    if toc_column and toc_column in result.columns:
        toc = to_fraction(result[toc_column], units["toc"])
        result["toc_clean"] = toc.fillna(0.0).clip(lower=0.0)

    # 第三步：提取矿物质量分数，并完成负值截断与归一化。
    mineral_map = columns["minerals"]
    missing_columns = [source for source in mineral_map.values() if source not in result.columns]
    if missing_columns:
        raise ValueError(f"Missing mineral columns: {missing_columns}")

    weight_fraction = pd.DataFrame(
        {mineral: to_fraction(result[source], units["mineral_fractions"]) for mineral, source in mineral_map.items()},
        index=result.index,
    )
    raw_weight_sum = weight_fraction.sum(axis=1)

    negative_count = (weight_fraction.fillna(0.0) < 0.0).sum()
    if preprocess_cfg["clip_negative"]:
        weight_fraction = weight_fraction.fillna(0.0).clip(lower=0.0)

    row_sum_before_norm = weight_fraction.sum(axis=1)
    if preprocess_cfg["normalize_minerals"]:
        weight_fraction = weight_fraction.div(row_sum_before_norm.where(row_sum_before_norm > 0.0), axis=0).fillna(0.0)

    # 第四步：把矿物质量分数转换成体积分数，为矿物混合计算做准备。
    density_map = {name: minerals_cfg[name]["density"] for name in mineral_map}
    volume_fraction, raw_volume_sum = weight_to_volume_fraction(weight_fraction, density_map)

    for mineral in mineral_map:
        result[f"{mineral}_wt_clean"] = weight_fraction[mineral]
        result[f"{mineral}_vol"] = volume_fraction[mineral]

    result["mineral_weight_sum_raw"] = raw_weight_sum
    result["mineral_row_sum_before_norm"] = row_sum_before_norm
    result["mineral_wt_clean_sum"] = weight_fraction.sum(axis=1)
    result["mineral_vol_sum"] = volume_fraction.sum(axis=1)
    result["mineral_grain_density"] = 1.0 / raw_volume_sum.where(raw_volume_sum > 0.0)
    result["mineral_valid"] = row_sum_before_norm > 0.0

    # 第五步：生成测井与建模质量控制标记，方便后面筛选有效样本。
    result["sonic_valid"] = (
        result["vp_log_m_s"].notna()
        & result["vs_log_m_s"].notna()
        & (result["vp_log_m_s"] > 0.0)
        & (result["vs_log_m_s"] > 0.0)
    )
    result["density_valid"] = result["rho_log_gcc"].notna() & (result["rho_log_gcc"] > 0.0)
    result["porosity_valid"] = result["phi"].notna() & (result["phi"] >= 0.0) & (result["phi"] < 1.0)
    result["sw_valid"] = result["sw"].notna() & (result["sw"] >= 0.0) & (result["sw"] <= 1.0)

    # 第六步：由实测速度和密度反算动态模量，作为模型标定参照。
    if preprocess_cfg["compute_log_moduli"]:
        result["K_log_GPa"], result["G_log_GPa"] = compute_log_moduli(
            result["vp_log_m_s"],
            result["vs_log_m_s"],
            result["rho_log_gcc"],
        )
        result["elastic_log_valid"] = (
            result["K_log_GPa"].notna()
            & result["G_log_GPa"].notna()
            & (result["K_log_GPa"] > 0.0)
            & (result["G_log_GPa"] > 0.0)
        )
    else:
        result["elastic_log_valid"] = False

    result["model_valid"] = result["mineral_valid"] & result["porosity_valid"] & result["sw_valid"]

    if preprocess_cfg["drop_invalid_rows"]:
        result = result.loc[result["model_valid"]].copy()

    summary = {
        "rows": int(len(result)),
        "mineral_invalid_rows": int((~result["mineral_valid"]).sum()),
        "sonic_invalid_rows": int((~result["sonic_valid"]).sum()),
        "density_invalid_rows": int((~result["density_valid"]).sum()),
        "porosity_invalid_rows": int((~result["porosity_valid"]).sum()),
        "sw_invalid_rows": int((~result["sw_valid"]).sum()),
    }
    for mineral in mineral_map:
        summary[f"{mineral}_negative_count"] = int(negative_count.get(mineral, 0))
    if "toc_clean" in result.columns:
        summary["toc_negative_count"] = int((to_fraction(result[toc_column], units["toc"]).fillna(0.0) < 0.0).sum())

    return result, summary
