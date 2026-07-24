"""命令行入口。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .conversion import ConversionError, convert_dataset, load_conversion_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="将 Piper HDF5 轨迹转换为 OpenPI 可读取的 LeRobot v2.1 数据集。")
    parser.add_argument("--config", type=Path, required=True, help="转换 YAML 配置路径")
    parser.add_argument("--max-episodes", type=int, help="仅转换排序后的前 N 条轨迹，用于小样本验证")
    parser.add_argument("--overwrite", action="store_true", help="显式覆盖 output.root 中已有的转换产物")
    parser.add_argument("--quiet", action="store_true", help="关闭逐轨迹与第三方进度日志，适合全量批处理")
    args = parser.parse_args(argv)
    try:
        config = load_conversion_config(args.config)
        if args.overwrite:
            config = replace(config, output=replace(config.output, overwrite=True))
        reports = convert_dataset(config, max_episodes=args.max_episodes, verbose=not args.quiet)
    except ConversionError as error:
        parser.error(str(error))
    print(json.dumps({"episodes": len(reports), "frames": sum(report.frames for report in reports)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
