from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from crossplotting import plot_crossplots
from fracture_analysis_v2 import run_fracture_analysis
from fracture_inversion import run_fracture_inversion
from plotting import plot_velocity_depth_panel
from preprocess_v2 import parse_sheet_name, preprocess_dataframe, resolve_path
from rp_model_v2 import run_model
from sensitivity_v2 import run_sensitivity_analysis


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_excel(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing and baseline rock physics modeling.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    # 第一步：读取配置文件，并解析输入输出路径。
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base_dir = config_path.parent

    input_path = resolve_path(base_dir, config["paths"]["input_file"])
    preprocessed_path = resolve_path(base_dir, config["paths"]["output_preprocessed"])
    modeled_path = resolve_path(base_dir, config["paths"]["output_modeled"])
    sheet_name = parse_sheet_name(config["paths"].get("sheet_name", 0))

    # 第二步：对原始测井和矿物组成数据做标准化前处理。
    frame = pd.read_excel(input_path, sheet_name=sheet_name)
    preprocessed, preprocess_summary = preprocess_dataframe(frame, config)
    save_excel(preprocessed, preprocessed_path)

    # 第三步：调用岩石物理建模模块，生成基线建模结果。
    modeled, model_summary = run_model(preprocessed, config, base_dir)
    save_excel(modeled, modeled_path)

    # 第四步：在基线模型上执行参数敏感性分析，输出参数扫描结果表和图件。
    sensitivity_result = run_sensitivity_analysis(preprocessed, config, base_dir)

    # 第五步：在无裂缝背景模型基础上执行裂缝密度-倾角正演敏感性分析。
    fracture_result = run_fracture_analysis(modeled, config, base_dir)

    # 第六步：执行裂缝密度和裂缝倾角联合反演，得到随深度变化的等效裂缝参数。
    fracture_inversion_result = run_fracture_inversion(modeled, config, base_dir)

    # 第七步：基于裂缝反演结果绘制页岩油工区解释所需的交会图。
    crossplot_path = None
    if fracture_inversion_result is not None:
        crossplot_path = plot_crossplots(fracture_inversion_result["result_frame"], config, base_dir)

    # 第八步：根据建模结果绘制速度与误差深度综合图。
    figure_path = plot_velocity_depth_panel(modeled, config, base_dir)

    # 第九步：输出本次运行摘要，便于快速检查结果。
    print(f"Input file: {input_path}")
    print(f"Preprocessed file: {preprocessed_path}")
    print(f"Modeled file: {modeled_path}")
    if sensitivity_result is not None:
        print(f"Sensitivity table: {sensitivity_result['output_table']}")
        print("Sensitivity summary:")
        for key, value in sensitivity_result["summary"].items():
            print(f"  {key}: {value}")
    if fracture_result is not None:
        if "reference_output" in fracture_result:
            print(f"Fracture reference file: {fracture_result['reference_output']}")
        if "output_table" in fracture_result:
            print(f"Fracture table: {fracture_result['output_table']}")
        if "grid_figure_path" in fracture_result:
            print(f"Fracture figure: {fracture_result['grid_figure_path']}")
        print("Fracture summary:")
        for key, value in fracture_result["summary"].items():
            print(f"  {key}: {value}")
    if fracture_inversion_result is not None:
        print(f"Fracture inversion file: {fracture_inversion_result['output_file']}")
        if fracture_inversion_result.get("figure_path") is not None:
            print(f"Fracture inversion figure: {fracture_inversion_result['figure_path']}")
        if fracture_inversion_result.get("velocity_panel_path") is not None:
            print(f"Fracture inversion velocity panel: {fracture_inversion_result['velocity_panel_path']}")
        print("Fracture inversion summary:")
        for key, value in fracture_inversion_result["summary"].items():
            print(f"  {key}: {value}")
    if crossplot_path is not None:
        print(f"Crossplot panel: {crossplot_path}")
    if figure_path is not None:
        print(f"Figure file: {figure_path}")
    print("Preprocess summary:")
    for key, value in preprocess_summary.items():
        print(f"  {key}: {value}")
    print("Model summary:")
    for key, value in model_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
