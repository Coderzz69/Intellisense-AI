#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the paper's pixel-to-meter curve y = alpha * x^n from calibration samples."
    )
    parser.add_argument("--input", required=True, help="CSV with columns: pixel_length,meter_length")
    parser.add_argument("--output", help="Optional JSON file for the fitted constants.")
    return parser.parse_args()


def read_samples(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    pixels: List[float] = []
    meters: List[float] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pixel = float(row["pixel_length"])
            meter = float(row["meter_length"])
            if pixel <= 0.0 or meter <= 0.0:
                continue
            pixels.append(pixel)
            meters.append(meter)
    if not pixels:
        raise ValueError("No valid calibration rows found.")
    return np.array(pixels), np.array(meters)


def main() -> None:
    args = parse_args()
    pixels, meters = read_samples(Path(args.input))
    exponent, log_alpha = np.polyfit(np.log(pixels), np.log(meters), deg=1)
    payload = {
        "alpha": float(np.exp(log_alpha)),
        "exponent": float(exponent),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
