# adjust limits of the analysis
def adjust_limits(df, col, clip="median", qty_std=2, quantile=(0.05, 0.95)):
    if clip == "median":
        return (
            max(df[col].median() - qty_std * df[col].std(), df[col].min()),
            min(df[col].median() + qty_std * df[col].std(), df[col].max()),
        )
    if clip == "quantile":
        return (
            max(df[col].quantile(quantile[0]), df[col].min()),
            min(df[col].quantile(quantile[1]), df[col].max()),
        )
    return clip


def get_limits(df, col, clip=None):
    if clip:
        return (max(clip[0], df[col].min()), min(clip[1], df[col].max()))
    else:
        return (df[col].min(), df[col].max())


def annotate(ax, format_num=".2f", height_min=0.02):
    for p in ax.patches:
        left, bottom, width, height = p.get_bbox().bounds
        annot = format(height, format_num)
        percentage = height / ax.get_ylim()[1]
        if percentage > height_min:
            ax.annotate(
                annot,
                (left + width / 2, bottom + height / 2),
                fontsize=min(max(percentage, 0.05), 0.08) * 200,
                ha="center",
                va="center",
            )
