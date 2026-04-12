from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_v2 import resolve_path
from rp_model_v2 import run_model


METRIC_INFO = {
    "vp_rmse": {"column": "vp_misfit_m_s", "label": "Vp RMSE (m/s)"},
    "vs_rmse": {"column": "vs_misfit_m_s", "label": "Vs RMSE (m/s)"},
    "rho_rmse": {"column": "rho_misfit_gcc", "label": "Density RMSE (g/cc)"},
    "vp_bias": {"column": "vp_misfit_m_s", "label": "Vp Bias (m/s)"},
    "vs_bias": {"column": "vs_misfit_m_s", "label": "Vs Bias (m/s)"},
    "composite_score": {"column": "", "label": "Composite Score"},
}


def _configure_matplotlib(base_dir: Path):
    mpl_config_dir = base_dir / ".mplconfig" / f"run_{os.getpid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    import matplotlib.pyplot as plt

    return plt


def get_nested_value(config: dict, path: str):
    current = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Config path not found: {path}")
        current = current[key]
    return current


def set_nested_value(config: dict, path: str, value) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"Config path not found: {path}")
        current = current[key]
    current[keys[-1]] = value


def sanitize_name(value: str) -> str:
    return value.replace(".", "_").replace("/", "_").replace("\\", "_")


def validate_metrics(metrics: list[str], allow_composite: bool = False) -> list[str]:
    valid_metrics = set(METRIC_INFO)
    if not allow_composite:
        valid_metrics.discard("composite_score")
    invalid = [metric for metric in metrics if metric not in valid_metrics]
    if invalid:
        raise ValueError(f"Unsupported sensitivity metrics: {invalid}")
    return metrics


def get_score_weights(config: dict) -> dict[str, float]:
    score_weights = config.get("sensitivity", {}).get("score_weights", {})
    if not score_weights:
        return {
            "vp_rmse": 1.0,
            "vs_rmse": 1.0,
            "rho_rmse": 0.5,
            "vp_bias": 0.5,
            "vs_bias": 0.5,
        }
    return {metric: float(weight) for metric, weight in score_weights.items()}


def compute_metrics(frame: pd.DataFrame, metrics: list[str]) -> dict[str, float | int]:
    compare = frame.loc[frame["compare_valid"]].copy()
    summary: dict[str, float | int] = {"compare_rows": int(len(compare))}
    if compare.empty:
        for metric in metrics:
            summary[metric] = np.nan
        return summary

    for metric in metrics:
        column = METRIC_INFO[metric]["column"]
        values = compare[column].to_numpy(dtype=float)
        if metric.endswith("_rmse"):
            summary[metric] = float(np.sqrt(np.nanmean(values**2)))
        elif metric.endswith("_bias"):
            summary[metric] = float(np.nanmean(values))
        else:
            raise ValueError(f"Unsupported sensitivity metric: {metric}")
    return summary


def score_result_frame(frame: pd.DataFrame, score_weights: dict[str, float]) -> pd.DataFrame:
    scored = frame.copy()
    total_weight = np.zeros(len(scored), dtype=float)
    weighted_sum = np.zeros(len(scored), dtype=float)

    # 第一步：对每个评分指标做归一化，便于把不同量纲的误差合成同一评分。
    for metric, weight in score_weights.items():
        if metric not in scored.columns:
            continue
        metric_values = scored[metric].to_numpy(dtype=float)
        if metric.endswith("_bias"):
            metric_values = np.abs(metric_values)

        normalized = np.full(len(scored), np.nan, dtype=float)
        valid = np.isfinite(metric_values)
        if valid.any():
            valid_values = metric_values[valid]
            lower = float(np.nanmin(valid_values))
            upper = float(np.nanmax(valid_values))
            if np.isclose(upper, lower):
                normalized[valid] = 0.0
            else:
                normalized[valid] = (valid_values - lower) / (upper - lower)

        scored[f"{metric}_score_component"] = normalized
        valid_component = np.isfinite(normalized)
        weighted_sum[valid_component] += weight * normalized[valid_component]
        total_weight[valid_component] += weight

    # 第二步：按权重求综合评分，分值越小表示整体拟合越好。
    composite_score = np.full(len(scored), np.nan, dtype=float)
    valid_score = total_weight > 0.0
    composite_score[valid_score] = weighted_sum[valid_score] / total_weight[valid_score]
    scored["composite_score"] = composite_score
    return scored


def scan_single_parameter(
    frame: pd.DataFrame,
    config: dict,
    base_dir: Path,
    parameter_path: str,
    values: list[float],
    metrics: list[str],
) -> pd.DataFrame:
    baseline_value = get_nested_value(config, parameter_path)
    rows = []

    # 第一步：保持其它参数不变，仅扫描当前一个参数，生成一维敏感性结果。
    for value in values:
        scan_config = copy.deepcopy(config)
        set_nested_value(scan_config, parameter_path, value)
        modeled, model_summary = run_model(frame, scan_config, base_dir)
        metric_summary = compute_metrics(modeled, metrics)

        rows.append(
            {
                "parameter": parameter_path,
                "value": value,
                "baseline_value": baseline_value,
                "modeled_rows": int(model_summary["modeled_rows"]),
                **metric_summary,
            }
        )

    return pd.DataFrame(rows).sort_values("value", kind="mergesort").reset_index(drop=True)


def scan_two_parameters(
    frame: pd.DataFrame,
    config: dict,
    base_dir: Path,
    x_parameter: str,
    y_parameter: str,
    x_values: list[float],
    y_values: list[float],
    metrics: list[str],
) -> pd.DataFrame:
    baseline_x = get_nested_value(config, x_parameter)
    baseline_y = get_nested_value(config, y_parameter)
    rows = []

    # 第一步：构建双参数网格，对每组参数组合都重新跑一次建模。
    for x_value in x_values:
        for y_value in y_values:
            scan_config = copy.deepcopy(config)
            set_nested_value(scan_config, x_parameter, x_value)
            set_nested_value(scan_config, y_parameter, y_value)
            modeled, model_summary = run_model(frame, scan_config, base_dir)
            metric_summary = compute_metrics(modeled, metrics)

            rows.append(
                {
                    "x_parameter": x_parameter,
                    "y_parameter": y_parameter,
                    "x_value": x_value,
                    "y_value": y_value,
                    "baseline_x": baseline_x,
                    "baseline_y": baseline_y,
                    "modeled_rows": int(model_summary["modeled_rows"]),
                    **metric_summary,
                }
            )

    return pd.DataFrame(rows).sort_values(["x_value", "y_value"], kind="mergesort").reset_index(drop=True)


def summarize_best_single_parameter(summary_frame: pd.DataFrame) -> pd.DataFrame:
    if "composite_score" not in summary_frame.columns:
        return pd.DataFrame()
    return (
        summary_frame.sort_values(["parameter", "composite_score", "vp_rmse", "vs_rmse"], kind="mergesort")
        .groupby("parameter", as_index=False)
        .first()
        .reset_index(drop=True)
    )


def summarize_parameter_ranges(summary_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for parameter, group in summary_frame.groupby("parameter"):
        rows.append(
            {
                "parameter": parameter,
                "composite_score_range": float(group["composite_score"].max() - group["composite_score"].min()),
                "vp_rmse_range": float(group["vp_rmse"].max() - group["vp_rmse"].min()),
                "vs_rmse_range": float(group["vs_rmse"].max() - group["vs_rmse"].min()),
                "rho_rmse_range": float(group["rho_rmse"].max() - group["rho_rmse"].min()) if "rho_rmse" in group else np.nan,
                "vp_bias_range": float(group["vp_bias"].max() - group["vp_bias"].min()),
                "vs_bias_range": float(group["vs_bias"].max() - group["vs_bias"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["composite_score_range", "vp_rmse_range", "vs_rmse_range"],
        ascending=False,
        kind="mergesort",
    ).reset_index(drop=True)


def build_recommendations(single_best: pd.DataFrame, grid_frame: pd.DataFrame | None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    if not single_best.empty:
        for _, row in single_best.iterrows():
            rows.append(
                {
                    "source": "single_parameter",
                    "parameter": row["parameter"],
                    "value": row["value"],
                    "composite_score": row["composite_score"],
                    "vp_rmse": row["vp_rmse"],
                    "vs_rmse": row["vs_rmse"],
                    "vp_bias": row["vp_bias"],
                    "vs_bias": row["vs_bias"],
                    "rho_rmse": row.get("rho_rmse", np.nan),
                }
            )

    if grid_frame is not None and not grid_frame.empty:
        best_grid = grid_frame.sort_values(["composite_score", "vp_rmse", "vs_rmse"], kind="mergesort").iloc[0]
        rows.append(
            {
                "source": "two_parameter_grid",
                "parameter": f"{best_grid['x_parameter']} | {best_grid['y_parameter']}",
                "value": f"{best_grid['x_value']} | {best_grid['y_value']}",
                "composite_score": best_grid["composite_score"],
                "vp_rmse": best_grid["vp_rmse"],
                "vs_rmse": best_grid["vs_rmse"],
                "vp_bias": best_grid["vp_bias"],
                "vs_bias": best_grid["vs_bias"],
                "rho_rmse": best_grid.get("rho_rmse", np.nan),
            }
        )

    return pd.DataFrame(rows)


def plot_single_parameter(
    result: pd.DataFrame,
    parameter_path: str,
    config: dict,
    base_dir: Path,
) -> Path:
    sensitivity_cfg = config["sensitivity"]
    metrics = [metric for metric in sensitivity_cfg["metrics"] if metric in result.columns]
    if "composite_score" in result.columns:
        metrics = metrics + ["composite_score"]

    plt = _configure_matplotlib(base_dir)

    output_dir = resolve_path(base_dir, sensitivity_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sensitivity_{sanitize_name(parameter_path)}.png"

    figure_width = float(sensitivity_cfg.get("figure_width", 16))
    figure_height = float(sensitivity_cfg.get("figure_height", 10))
    dpi = int(sensitivity_cfg.get("dpi", 300))
    axis_label_fontsize = float(sensitivity_cfg.get("axis_label_fontsize", 14))
    tick_label_fontsize = float(sensitivity_cfg.get("tick_label_fontsize", 12))
    title_fontsize = float(sensitivity_cfg.get("title_fontsize", 16))
    line_width = float(sensitivity_cfg.get("line_width", 1.8))
    marker_size = float(sensitivity_cfg.get("marker_size", 6))

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, figure_height), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    x = result["value"].to_numpy(dtype=float)
    baseline_value = float(result["baseline_value"].iloc[0])

    # 第一步：逐个指标绘制参数值与误差统计量之间的关系曲线。
    for axis, metric in zip(axes, metrics):
        y = result[metric].to_numpy(dtype=float)
        axis.plot(x, y, marker="o", lw=line_width, ms=marker_size, color="tab:blue")
        axis.axvline(baseline_value, color="black", lw=1.0, linestyle="--")
        if metric.endswith("_bias"):
            axis.axhline(0.0, color="gray", lw=1.0, linestyle=":")
        axis.set_title(METRIC_INFO[metric]["label"], fontsize=title_fontsize)
        axis.set_xlabel("Parameter Value", fontsize=axis_label_fontsize)
        axis.set_ylabel(METRIC_INFO[metric]["label"], fontsize=axis_label_fontsize)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize)

    # 第二步：关闭多余子图，并把参数名作为整张图的总标题。
    for axis in axes[len(metrics) :]:
        axis.set_visible(False)
    fig.suptitle(parameter_path, fontsize=title_fontsize + 1)

    # 第三步：保存图件并释放内存。
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_two_parameter_grid(
    result: pd.DataFrame,
    config: dict,
    base_dir: Path,
    metrics: list[str],
) -> Path:
    sensitivity_cfg = config["sensitivity"]
    grid_cfg = sensitivity_cfg["two_parameter_grid"]
    plt = _configure_matplotlib(base_dir)

    output_dir = resolve_path(base_dir, sensitivity_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"grid_{sanitize_name(grid_cfg['x_parameter'])}_{sanitize_name(grid_cfg['y_parameter'])}.png"
    )

    figure_width = float(grid_cfg.get("figure_width", sensitivity_cfg.get("figure_width", 16)))
    figure_height = float(grid_cfg.get("figure_height", sensitivity_cfg.get("figure_height", 10)))
    dpi = int(grid_cfg.get("dpi", sensitivity_cfg.get("dpi", 300)))
    axis_label_fontsize = float(grid_cfg.get("axis_label_fontsize", sensitivity_cfg.get("axis_label_fontsize", 14)))
    tick_label_fontsize = float(grid_cfg.get("tick_label_fontsize", sensitivity_cfg.get("tick_label_fontsize", 12)))
    title_fontsize = float(grid_cfg.get("title_fontsize", sensitivity_cfg.get("title_fontsize", 16)))

    x_parameter = str(grid_cfg["x_parameter"])
    y_parameter = str(grid_cfg["y_parameter"])
    best_row = result.sort_values(["composite_score", "vp_rmse", "vs_rmse"], kind="mergesort").iloc[0]

    ncols = 2
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figure_width, figure_height), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    # 第一步：把双参数扫描结果转成网格矩阵，并为每个指标绘制热图。
    for axis, metric in zip(axes, metrics):
        pivot = (
            result.pivot(index="y_value", columns="x_value", values=metric)
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(METRIC_INFO[metric]["label"], fontsize=title_fontsize)
        axis.set_xlabel(x_parameter, fontsize=axis_label_fontsize)
        axis.set_ylabel(y_parameter, fontsize=axis_label_fontsize)
        axis.set_xticks(range(len(pivot.columns)))
        axis.set_xticklabels([f"{value:.3g}" for value in pivot.columns], fontsize=tick_label_fontsize)
        axis.set_yticks(range(len(pivot.index)))
        axis.set_yticklabels([f"{value:.3g}" for value in pivot.index], fontsize=tick_label_fontsize)

        best_x_index = list(pivot.columns).index(best_row["x_value"])
        best_y_index = list(pivot.index).index(best_row["y_value"])
        axis.scatter(best_x_index, best_y_index, marker="*", s=180, color="white", edgecolor="black")
        fig.colorbar(image, ax=axis, shrink=0.9)

    # 第二步：关闭多余子图，并在图中标记综合评分最优的参数组合。
    for axis in axes[len(metrics) :]:
        axis.set_visible(False)
    fig.suptitle(
        f"{x_parameter} x {y_parameter} best=({best_row['x_value']}, {best_row['y_value']})",
        fontsize=title_fontsize + 1,
    )

    # 第三步：保存二维敏感性热图。
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_sensitivity_analysis(frame: pd.DataFrame, config: dict, base_dir: Path) -> dict | None:
    sensitivity_cfg = config.get("sensitivity", {})
    if not sensitivity_cfg.get("enabled", False):
        return None

    metrics = validate_metrics(list(sensitivity_cfg.get("metrics", [])))
    score_weights = get_score_weights(config)
    parameter_map = sensitivity_cfg.get("parameters", {})
    if not parameter_map:
        raise ValueError("sensitivity.parameters is empty.")

    all_results = []
    figure_paths: list[Path] = []

    # 第一步：对配置中给定的每个参数逐一做一维扫描，并汇总误差指标。
    for parameter_path, values in parameter_map.items():
        if not values:
            continue
        result = scan_single_parameter(
            frame=frame,
            config=config,
            base_dir=base_dir,
            parameter_path=parameter_path,
            values=list(values),
            metrics=metrics,
        )
        scored_result = score_result_frame(result, score_weights)
        all_results.append(scored_result)
        figure_paths.append(plot_single_parameter(scored_result, parameter_path, config, base_dir))

    if not all_results:
        raise ValueError("No sensitivity results were generated.")

    summary_frame = pd.concat(all_results, ignore_index=True)
    single_best = summarize_best_single_parameter(summary_frame)
    parameter_ranges = summarize_parameter_ranges(summary_frame)

    grid_frame: pd.DataFrame | None = None
    grid_figure_path: Path | None = None
    grid_cfg = sensitivity_cfg.get("two_parameter_grid", {})

    # 第二步：对最敏感的双参数组合做网格扫描，并绘制二维热图。
    if grid_cfg.get("enabled", False):
        x_parameter = str(grid_cfg["x_parameter"])
        y_parameter = str(grid_cfg["y_parameter"])
        x_values = list(grid_cfg.get("x_values", parameter_map.get(x_parameter, [])))
        y_values = list(grid_cfg.get("y_values", parameter_map.get(y_parameter, [])))
        if not x_values or not y_values:
            raise ValueError("two_parameter_grid requires x_values and y_values.")

        grid_frame = scan_two_parameters(
            frame=frame,
            config=config,
            base_dir=base_dir,
            x_parameter=x_parameter,
            y_parameter=y_parameter,
            x_values=x_values,
            y_values=y_values,
            metrics=metrics,
        )
        grid_frame = score_result_frame(grid_frame, score_weights)
        grid_metrics = validate_metrics(
            list(grid_cfg.get("metrics", ["composite_score", "vp_rmse", "vs_rmse"])),
            allow_composite=True,
        )
        grid_figure_path = plot_two_parameter_grid(grid_frame, config, base_dir, grid_metrics)
        figure_paths.append(grid_figure_path)

    recommendations = build_recommendations(single_best, grid_frame)

    # 第三步：把一维扫描、推荐结果和二维网格统一写入同一个结果工作簿。
    output_table = resolve_path(base_dir, sensitivity_cfg["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_table) as writer:
        summary_frame.to_excel(writer, sheet_name="single_parameter", index=False)
        single_best.to_excel(writer, sheet_name="best_by_parameter", index=False)
        parameter_ranges.to_excel(writer, sheet_name="parameter_ranges", index=False)
        if grid_frame is not None:
            grid_frame.to_excel(writer, sheet_name="two_parameter_grid", index=False)
        recommendations.to_excel(writer, sheet_name="recommendations", index=False)

    summary: dict[str, object] = {
        "parameter_count": len(all_results),
        "scan_rows": int(len(summary_frame)),
    }
    if not summary_frame.empty:
        overall_best_single = summary_frame.sort_values(
            ["composite_score", "vp_rmse", "vs_rmse"],
            kind="mergesort",
        ).iloc[0]
        summary["best_single_parameter"] = f"{overall_best_single['parameter']}={overall_best_single['value']}"
    if not recommendations.empty:
        best_grid = recommendations.loc[recommendations["source"] == "two_parameter_grid"].head(1)
        if not best_grid.empty:
            summary["best_grid_combo"] = str(best_grid.iloc[0]["value"])
    if not parameter_ranges.empty:
        summary["most_sensitive_parameter"] = str(parameter_ranges.iloc[0]["parameter"])
    if grid_frame is not None:
        summary["grid_rows"] = int(len(grid_frame))

    return {
        "output_table": output_table,
        "figure_paths": figure_paths,
        "grid_figure_path": grid_figure_path,
        "summary": summary,
    }
