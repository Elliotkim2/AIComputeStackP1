"""Print report-ready Markdown tables from p1_kv_probe.py's _kv.csv output.

Usage: python format_kv_table.py p1_results_kv.csv
       python format_kv_table.py p1_results_kv.csv > byte_tables.md
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


BYTE_COLUMNS = (
    "predicted_kv_bytes",
    "measured_kv_bytes",
    "logits_bytes",
    "other_output_bytes",
)
REQUIRED = {"model", "seq_len", *BYTE_COLUMNS}
MIB = 1024 ** 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", type=Path, help="The probe's *_kv.csv file")
    args = parser.parse_args()

    with args.csv_file.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            parser.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        by_model = defaultdict(list)
        for line_number, row in enumerate(reader, start=2):
            try:
                n = int(row["seq_len"])
                values = [int(row[column]) for column in BYTE_COLUMNS]
                model = row["model"].strip()
                if not model or n <= 0 or values[0] <= 0:
                    raise ValueError("empty model, nonpositive n, or nonpositive predicted KV")
            except (ValueError, TypeError) as exc:
                parser.error(f"invalid data on CSV line {line_number}: {exc}")
            by_model[model].append((n, values))

    if not by_model:
        parser.error("CSV has no data rows")

    for model, rows in by_model.items():
        print(f"### {model}\n")
        print("| n (tokens) | Predicted KV (MiB) | Measured KV (MiB) | "
              "Logits (MiB) | Other outputs (MiB) | Measured / predicted |")
        print("| ---: | ---: | ---: | ---: | ---: | ---: |")
        for n, values in sorted(rows):
            predicted, measured, logits, other = values
            fields = [str(n), *(f"{value / MIB:.2f}" for value in values),
                      f"{measured / predicted:.3f}"]
            print("| " + " | ".join(fields) + " |")
        print()


if __name__ == "__main__":
    main()
