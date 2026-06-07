import argparse
import sys
 
import pandas as pd
import matplotlib.pyplot as plt
 
 
def load_results(path: str, sheet: str = "Sheet1") -> pd.DataFrame:
    """Read the results table, tolerating leading empty rows/columns.
 
    The data in the source file starts at row 5 and column E, so the header
    row is located dynamically (the row containing "Model") and any fully
    empty rows/columns are dropped. This keeps the script working even if the
    table is moved around the sheet.
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
 
    # Find the header row: the first row containing the cell "Model".
    is_header = raw.apply(
        lambda row: row.astype(str).str.strip().eq("Model").any(), axis=1
    )
    if not is_header.any():
        raise ValueError('Could not find a header row containing "Model".')
    header_row = int(is_header.idxmax())
 
    # Re-read using that row as the header, then drop empty rows/columns.
    df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df.reset_index(drop=True)
 
 
def make_scatter(df: pd.DataFrame, output: str) -> None:
    """Plot the first metric column (x) against the second (y)."""
    label_col = df.columns[0]                       # "Model"
    metric_cols = list(df.columns[1:])              # ["Pixel Acc.", "MIoU"]
    if len(metric_cols) < 2:
        raise ValueError(
            f"Expected at least two metric columns, found: {metric_cols}"
        )
    x_col, y_col = metric_cols[0], metric_cols[1]
 
    # Ensure the metrics are numeric and drop rows missing either value.
    df = df.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=[x_col, y_col])
 
    # Separate "our" model so it can be highlighted.
    is_ours = df[label_col].astype(str).str.contains("ours", case=False)
    others, ours = df[~is_ours], df[is_ours]
 
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(
        others[x_col], others[y_col],
        s=90, color="#4C72B0", edgecolor="white", linewidth=0.8,
        zorder=3, label="Baselines",
    )
    if not ours.empty:
        ax.scatter(
            ours[x_col], ours[y_col],
            s=200, marker="*", color="#C44E52", edgecolor="white",
            linewidth=1.0, zorder=4, label="Ours",
        )
 
    # Annotate every point with its model name. When the optional adjustText
    # package is installed, labels are automatically repelled so they don't
    # overlap in dense clusters; otherwise fall back to simple offset labels.
    try:
        from adjustText import adjust_text
 
        texts = [
            ax.text(row[x_col], row[y_col], str(row[label_col]),
                    fontsize=9, zorder=5)
            for _, row in df.iterrows()
        ]
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
        )
    except ImportError:
        for _, row in df.iterrows():
            ax.annotate(
                str(row[label_col]),
                (row[x_col], row[y_col]),
                textcoords="offset points", xytext=(7, 5),
                fontsize=9, zorder=5,
            )
 
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f"{y_col} vs. {x_col} by Model", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.legend(frameon=True)
    ax.margins(0.12)  # a little padding so labels don't get clipped
    fig.tight_layout()
 
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved scatter plot to {output}")
 
 
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a scatter plot from experimental_results.xlsx"
    )
    parser.add_argument(
        "--input", default="experimental_results.xlsx",
        help="Path to the Excel file (default: experimental_results.xlsx)",
    )
    parser.add_argument(
        "--output", default="scatter_plot.png",
        help="Path for the output image (default: scatter_plot.png)",
    )
    parser.add_argument(
        "--sheet", default="Sheet1",
        help="Worksheet name to read (default: Sheet1)",
    )
    args = parser.parse_args()
 
    df = load_results(args.input, sheet=args.sheet)
    make_scatter(df, args.output)
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())