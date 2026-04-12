from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from fracture_analysis_v2 import compute_fracture_velocities
from preprocess_v2 import resolve_path


def _configure_matplotlib(base_dir: Path):
    mpl_config_dir = base_dir / ".mplconfig" / f"run_{os.getpid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

    import matplotlib

    matplotlib.use("Agg")
    matplotlib.set_loglevel("error")
    import matplotlib.pyplot as plt

    return plt


def build_state_grid(inversion_cfg: dict) -> list[tuple[float, float]]:
    density_grid = sorted({float(value) for value in inversion_cfg["density_grid"]})
    dip_grid = sorted({float(value) for value in inversion_cfg["dip_grid_deg"]})
    return [(density, dip_angle_deg) for density in density_grid for dip_angle_deg in dip_grid]


def build_working_frame(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame["compare_valid"].to_numpy(dtype=bool)
    working = frame.loc[valid].copy()
    working["original_index"] = working.index.to_numpy()
    working = working.sort_values("depth_m", kind="mergesort").reset_index(drop=True)
    return working


def compute_state_predictions(
    working: pd.DataFrame,
    config: dict,
    base_dir: Path,
    states: list[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_depth = len(working)
    n_state = len(states)
    vp_matrix = np.full((n_depth, n_state), np.nan)
    vs_matrix = np.full((n_depth, n_state), np.nan)
    valid_matrix = np.zeros((n_depth, n_state), dtype=bool)

    # 第一步：对每组候选裂缝密度和倾角正演一次，形成逐深度的速度预测矩阵。
    for state_index, (density, dip_angle_deg) in enumerate(states):
        velocities = compute_fracture_velocities(working, config, base_dir, density, dip_angle_deg)
        vp = velocities["vp_fracture_m_s"]
        vs = velocities["vsv_fracture_m_s"]
        valid = velocities["fracture_valid"] & np.isfinite(vp) & np.isfinite(vs)
        vp_matrix[:, state_index] = vp
        vs_matrix[:, state_index] = vs
        valid_matrix[:, state_index] = valid

    return vp_matrix, vs_matrix, valid_matrix


def compute_data_cost(
    working: pd.DataFrame,
    vp_matrix: np.ndarray,
    vs_matrix: np.ndarray,
    valid_matrix: np.ndarray,
    inversion_cfg: dict,
) -> np.ndarray:
    vp_log = working["vp_log_m_s"].to_numpy(dtype=float)
    vs_log = working["vs_log_m_s"].to_numpy(dtype=float)

    vp_weight = float(inversion_cfg.get("vp_weight", 1.0))
    vs_weight = float(inversion_cfg.get("vs_weight", 1.0))
    use_relative_error = bool(inversion_cfg.get("use_relative_error", True))

    vp_error = vp_matrix - vp_log[:, None]
    vs_error = vs_matrix - vs_log[:, None]
    if use_relative_error:
        vp_scale = np.where(np.abs(vp_log) > 1e-8, np.abs(vp_log), np.nan)[:, None]
        vs_scale = np.where(np.abs(vs_log) > 1e-8, np.abs(vs_log), np.nan)[:, None]
        vp_error = vp_error / vp_scale
        vs_error = vs_error / vs_scale

    cost = vp_weight * vp_error**2 + vs_weight * vs_error**2
    cost[~valid_matrix] = np.inf
    cost[~np.isfinite(cost)] = np.inf
    return cost


def build_transition_penalty(states: list[tuple[float, float]], inversion_cfg: dict) -> np.ndarray:
    densities = np.array([state[0] for state in states], dtype=float)
    dips = np.array([state[1] for state in states], dtype=float)

    density_diff = np.abs(densities[:, None] - densities[None, :])
    dip_diff = np.abs(dips[:, None] - dips[None, :])

    density_span = max(float(densities.max() - densities.min()), 1e-6)
    dip_span = max(float(dips.max() - dips.min()), 1e-6)

    lambda_density = float(inversion_cfg.get("smooth_lambda_density", 0.0))
    lambda_dip = float(inversion_cfg.get("smooth_lambda_dip", 0.0))
    penalty = lambda_density * (density_diff / density_span) ** 2 + lambda_dip * (dip_diff / dip_span) ** 2

    max_density_change = inversion_cfg.get("max_density_change")
    if max_density_change is not None:
        penalty[density_diff > float(max_density_change)] = np.inf

    max_dip_change_deg = inversion_cfg.get("max_dip_change_deg")
    if max_dip_change_deg is not None:
        penalty[dip_diff > float(max_dip_change_deg)] = np.inf

    return penalty


def solve_segment_path(data_cost: np.ndarray, transition_penalty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_depth, n_state = data_cost.shape
    cumulative = np.full((n_depth, n_state), np.inf)
    previous = np.full((n_depth, n_state), -1, dtype=int)

    first_valid = np.isfinite(data_cost[0])
    cumulative[0, first_valid] = data_cost[0, first_valid]
    if not first_valid.any():
        return np.full(n_depth, -1, dtype=int), np.full(n_depth, np.nan), np.full(n_depth, np.nan)

    # 第二步：用动态规划在全井上寻找一条总误差最小且变化平滑的裂缝参数路径。
    for depth_index in range(1, n_depth):
        previous_total = cumulative[depth_index - 1][:, None] + transition_penalty
        best_previous = np.argmin(previous_total, axis=0)
        best_previous_cost = previous_total[best_previous, np.arange(n_state)]
        valid_state = np.isfinite(data_cost[depth_index]) & np.isfinite(best_previous_cost)
        cumulative[depth_index, valid_state] = data_cost[depth_index, valid_state] + best_previous_cost[valid_state]
        previous[depth_index, valid_state] = best_previous[valid_state]

    if not np.isfinite(cumulative[-1]).any():
        return np.full(n_depth, -1, dtype=int), np.full(n_depth, np.nan), np.full(n_depth, np.nan)

    state_path = np.full(n_depth, -1, dtype=int)
    state_path[-1] = int(np.argmin(cumulative[-1]))
    for depth_index in range(n_depth - 1, 0, -1):
        state_path[depth_index - 1] = previous[depth_index, state_path[depth_index]]

    local_cost = data_cost[np.arange(n_depth), state_path]
    cumulative_cost = cumulative[np.arange(n_depth), state_path]
    return state_path, local_cost, cumulative_cost


def solve_path_with_gaps(data_cost: np.ndarray, transition_penalty: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_depth = data_cost.shape[0]
    state_path = np.full(n_depth, -1, dtype=int)
    local_cost = np.full(n_depth, np.nan)
    cumulative_cost = np.full(n_depth, np.nan)
    valid_rows = np.isfinite(data_cost).any(axis=1)

    segment_start = None
    for depth_index, is_valid in enumerate(valid_rows):
        if is_valid and segment_start is None:
            segment_start = depth_index
        if segment_start is not None and ((not is_valid) or depth_index == n_depth - 1):
            segment_end = depth_index if not is_valid else depth_index + 1
            segment_path, segment_local, segment_cumulative = solve_segment_path(
                data_cost[segment_start:segment_end],
                transition_penalty,
            )
            state_path[segment_start:segment_end] = segment_path
            local_cost[segment_start:segment_end] = segment_local
            cumulative_cost[segment_start:segment_end] = segment_cumulative
            segment_start = None

    return state_path, local_cost, cumulative_cost


def build_inversion_result(
    frame: pd.DataFrame,
    working: pd.DataFrame,
    states: list[tuple[float, float]],
    state_path: np.ndarray,
    local_cost: np.ndarray,
    cumulative_cost: np.ndarray,
    vp_matrix: np.ndarray,
    vs_matrix: np.ndarray,
    transition_penalty: np.ndarray,
) -> pd.DataFrame:
    result = frame.copy()
    result["fracture_inversion_valid"] = False
    result["fracture_density_inv"] = np.nan
    result["fracture_dip_angle_inv_deg"] = np.nan
    result["vp_fracture_inv_m_s"] = np.nan
    result["vs_fracture_inv_m_s"] = np.nan
    result["vp_fracture_inv_misfit_m_s"] = np.nan
    result["vs_fracture_inv_misfit_m_s"] = np.nan
    result["fracture_data_objective"] = np.nan
    result["fracture_path_objective"] = np.nan
    result["fracture_transition_penalty"] = np.nan
    result["fracture_dip_on_boundary"] = False

    selected = state_path >= 0
    if not np.any(selected):
        return result

    selected_rows = np.where(selected)[0]
    selected_states = state_path[selected_rows]

    density_values = np.array([state[0] for state in states], dtype=float)
    dip_values = np.array([state[1] for state in states], dtype=float)
    min_dip = float(dip_values.min())
    max_dip = float(dip_values.max())

    working_result = pd.DataFrame(
        {
            "original_index": working.loc[selected_rows, "original_index"].to_numpy(dtype=int),
            "fracture_inversion_valid": True,
            "fracture_density_inv": density_values[selected_states],
            "fracture_dip_angle_inv_deg": dip_values[selected_states],
            "vp_fracture_inv_m_s": vp_matrix[selected_rows, selected_states],
            "vs_fracture_inv_m_s": vs_matrix[selected_rows, selected_states],
            "fracture_data_objective": local_cost[selected_rows],
            "fracture_path_objective": cumulative_cost[selected_rows],
        }
    )
    working_result["vp_fracture_inv_misfit_m_s"] = working_result["vp_fracture_inv_m_s"] - working.loc[selected_rows, "vp_log_m_s"].to_numpy(dtype=float)
    working_result["vs_fracture_inv_misfit_m_s"] = working_result["vs_fracture_inv_m_s"] - working.loc[selected_rows, "vs_log_m_s"].to_numpy(dtype=float)
    working_result["fracture_dip_on_boundary"] = (
        np.isclose(working_result["fracture_dip_angle_inv_deg"], min_dip)
        | np.isclose(working_result["fracture_dip_angle_inv_deg"], max_dip)
    )

    transition_values = np.zeros(len(selected_rows), dtype=float)
    for offset in range(1, len(selected_rows)):
        prev_state = selected_states[offset - 1]
        curr_state = selected_states[offset]
        transition_values[offset] = transition_penalty[prev_state, curr_state]
    working_result["fracture_transition_penalty"] = transition_values

    result.loc[working_result["original_index"], working_result.columns.drop("original_index")] = working_result.drop(
        columns="original_index"
    ).to_numpy()
    return result


def plot_inversion_panel(result: pd.DataFrame, config: dict, base_dir: Path) -> Path | None:
    inversion_cfg = config["fracture"]["inversion"]
    output_figure = inversion_cfg.get("output_figure")
    if not output_figure:
        return None

    plot_data = result.loc[result["fracture_inversion_valid"]].copy()
    if plot_data.empty:
        return None

    plot_data = plot_data.sort_values("depth_m", kind="mergesort")
    plt = _configure_matplotlib(base_dir)
    output_path = resolve_path(base_dir, output_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_width = float(inversion_cfg.get("figure_width", 20))
    figure_height = float(inversion_cfg.get("figure_height", 12))
    dpi = int(inversion_cfg.get("dpi", 300))
    axis_label_fontsize = float(inversion_cfg.get("axis_label_fontsize", 16))
    tick_label_fontsize = float(inversion_cfg.get("tick_label_fontsize", 14))
    title_fontsize = float(inversion_cfg.get("title_fontsize", 18))
    legend_fontsize = float(inversion_cfg.get("legend_fontsize", 13))
    line_width = float(inversion_cfg.get("line_width", 1.8))

    depth = plot_data["depth_m"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(figure_width, figure_height), sharey=True, constrained_layout=True)

    # 第三步：把反演得到的裂缝密度、倾角以及反演后速度结果统一画成深度综合图。
    axes[0].plot(plot_data["fracture_density_inv"], depth, lw=line_width, color="tab:purple")
    axes[0].set_title("Fracture Density", fontsize=title_fontsize)
    axes[0].set_xlabel("Density", fontsize=axis_label_fontsize)
    axes[0].set_ylabel("Depth (m)", fontsize=axis_label_fontsize)

    axes[1].plot(plot_data["fracture_dip_angle_inv_deg"], depth, lw=line_width, color="tab:green")
    axes[1].set_title("Fracture Dip", fontsize=title_fontsize)
    axes[1].set_xlabel("Dip (deg)", fontsize=axis_label_fontsize)

    axes[2].plot(plot_data["vp_log_m_s"], depth, lw=line_width, label="Vp_log")
    axes[2].plot(plot_data["vp_fracture_inv_m_s"], depth, lw=line_width, label="Vp_inv")
    axes[2].set_title("Vp", fontsize=title_fontsize)
    axes[2].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[2].legend(fontsize=legend_fontsize)

    axes[3].plot(plot_data["vs_log_m_s"], depth, lw=line_width, label="Vs_log")
    axes[3].plot(plot_data["vs_fracture_inv_m_s"], depth, lw=line_width, label="Vs_inv")
    axes[3].set_title("Vs", fontsize=title_fontsize)
    axes[3].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[3].legend(fontsize=legend_fontsize)

    axes[0].set_ylim(float(depth.max()), float(depth.min()))
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_inversion_velocity_panel(result: pd.DataFrame, config: dict, base_dir: Path) -> Path | None:
    inversion_cfg = config["fracture"]["inversion"]
    output_figure = inversion_cfg.get("output_velocity_panel")
    if not output_figure:
        return None

    plot_data = result.loc[result["fracture_inversion_valid"]].copy()
    if plot_data.empty:
        return None

    plot_data = plot_data.sort_values("depth_m", kind="mergesort")
    plt = _configure_matplotlib(base_dir)
    output_path = resolve_path(base_dir, output_figure)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure_width = float(inversion_cfg.get("figure_width", 20))
    figure_height = float(inversion_cfg.get("figure_height", 12))
    dpi = int(inversion_cfg.get("dpi", 300))
    axis_label_fontsize = float(inversion_cfg.get("axis_label_fontsize", 16))
    tick_label_fontsize = float(inversion_cfg.get("tick_label_fontsize", 14))
    title_fontsize = float(inversion_cfg.get("title_fontsize", 18))
    legend_fontsize = float(inversion_cfg.get("legend_fontsize", 13))
    line_width = float(inversion_cfg.get("line_width", 1.8))

    depth = plot_data["depth_m"].to_numpy(dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(figure_width, figure_height), sharey=True, constrained_layout=True)

    # 第四步：绘制裂缝反演后的速度与误差深度综合图，便于和无裂缝基线图直接对照。
    axes[0].plot(plot_data["vp_log_m_s"], depth, label="Vp_log", lw=line_width)
    axes[0].plot(plot_data["vp_fracture_inv_m_s"], depth, label="Vp_inv", lw=line_width)
    axes[0].set_title("Vp", fontsize=title_fontsize)
    axes[0].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[0].set_ylabel("Depth (m)", fontsize=axis_label_fontsize)
    axes[0].legend(fontsize=legend_fontsize)

    axes[1].plot(plot_data["vs_log_m_s"], depth, label="Vs_log", lw=line_width)
    axes[1].plot(plot_data["vs_fracture_inv_m_s"], depth, label="Vs_inv", lw=line_width)
    axes[1].set_title("Vs", fontsize=title_fontsize)
    axes[1].set_xlabel("Velocity (m/s)", fontsize=axis_label_fontsize)
    axes[1].legend(fontsize=legend_fontsize)

    axes[2].plot(plot_data["vp_fracture_inv_misfit_m_s"], depth, label="Vp_error", lw=line_width, color="tab:red")
    axes[2].axvline(0.0, color="black", lw=1.0, linestyle="--")
    axes[2].set_title("Vp Error", fontsize=title_fontsize)
    axes[2].set_xlabel("Error (m/s)", fontsize=axis_label_fontsize)
    axes[2].legend(fontsize=legend_fontsize)

    axes[3].plot(plot_data["vs_fracture_inv_misfit_m_s"], depth, label="Vs_error", lw=line_width, color="tab:orange")
    axes[3].axvline(0.0, color="black", lw=1.0, linestyle="--")
    axes[3].set_title("Vs Error", fontsize=title_fontsize)
    axes[3].set_xlabel("Error (m/s)", fontsize=axis_label_fontsize)
    axes[3].legend(fontsize=legend_fontsize)

    axes[0].set_ylim(float(depth.max()), float(depth.min()))
    for axis in axes:
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.tick_params(axis="both", labelsize=tick_label_fontsize)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_fracture_inversion(frame: pd.DataFrame, config: dict, base_dir: Path) -> dict | None:
    fracture_cfg = config.get("fracture", {})
    inversion_cfg = fracture_cfg.get("inversion", {})
    if not fracture_cfg.get("enabled", False) or not inversion_cfg.get("enabled", False):
        return None

    working = build_working_frame(frame)
    if working.empty:
        raise ValueError("No valid rows available for fracture inversion.")

    states = build_state_grid(inversion_cfg)
    if not states:
        raise ValueError("fracture.inversion requires non-empty density_grid and dip_grid_deg.")

    vp_matrix, vs_matrix, valid_matrix = compute_state_predictions(working, config, base_dir, states)
    data_cost = compute_data_cost(working, vp_matrix, vs_matrix, valid_matrix, inversion_cfg)
    transition_penalty = build_transition_penalty(states, inversion_cfg)
    state_path, local_cost, cumulative_cost = solve_path_with_gaps(data_cost, transition_penalty)
    result = build_inversion_result(
        frame=frame,
        working=working,
        states=states,
        state_path=state_path,
        local_cost=local_cost,
        cumulative_cost=cumulative_cost,
        vp_matrix=vp_matrix,
        vs_matrix=vs_matrix,
        transition_penalty=transition_penalty,
    )

    output_file = resolve_path(base_dir, inversion_cfg["output_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_file, index=False)

    figure_path = plot_inversion_panel(result, config, base_dir)
    velocity_panel_path = plot_inversion_velocity_panel(result, config, base_dir)

    valid_result = result.loc[result["fracture_inversion_valid"]].copy()
    summary = {
        "inversion_rows": int(len(valid_result)),
        "state_count": int(len(states)),
    }
    if not valid_result.empty:
        summary["mean_fracture_density"] = float(valid_result["fracture_density_inv"].mean())
        summary["min_fracture_density"] = float(valid_result["fracture_density_inv"].min())
        summary["max_fracture_density"] = float(valid_result["fracture_density_inv"].max())
        summary["mean_fracture_dip_deg"] = float(valid_result["fracture_dip_angle_inv_deg"].mean())
        summary["min_fracture_dip_deg"] = float(valid_result["fracture_dip_angle_inv_deg"].min())
        summary["max_fracture_dip_deg"] = float(valid_result["fracture_dip_angle_inv_deg"].max())
        summary["boundary_dip_rows"] = int(valid_result["fracture_dip_on_boundary"].sum())

    return {
        "output_file": output_file,
        "figure_path": figure_path,
        "velocity_panel_path": velocity_panel_path,
        "result_frame": result,
        "summary": summary,
    }
