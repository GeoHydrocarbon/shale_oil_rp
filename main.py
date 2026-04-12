from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from plotting import plot_velocity_depth_panel
from preprocess_v2 import parse_sheet_name, preprocess_dataframe, resolve_path
from rp_model_v2 import run_model


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

    # 第三步：调用岩石物理模型模块，生成基线建模结果。
    modeled, model_summary = run_model(preprocessed, config, base_dir)
    save_excel(modeled, modeled_path)

    # 第四步：根据建模结果绘制速度与误差深度综合图。
    figure_path = plot_velocity_depth_panel(modeled, config, base_dir)

    # 第五步：输出本次运行摘要，便于快速检查结果。
    print(f"Input file: {input_path}")
    print(f"Preprocessed file: {preprocessed_path}")
    print(f"Modeled file: {modeled_path}")
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
