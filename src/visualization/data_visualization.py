import math
import os
import warnings
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_metric.functions import BinaryClassification
from scipy import stats
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (DetCurveDisplay, classification_report,
                             confusion_matrix, det_curve, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, precision_recall_curve,
                             r2_score)
from yellowbrick.bestfit import draw_identity_line
from yellowbrick.regressor import PredictionError, ResidualsPlot

from ..utils.utils import check_dict_keys
from .utils.utils import adjust_limits, annotate, get_limits

warnings.filterwarnings("ignore")

sns.set_palette(sns.cubehelix_palette(6, start=0.45, rot=-1.75, gamma=2, hue=1.25))
# sns.set_palette('Set2',8)
# sns_palette=color_palette(sns.cubehelix_palette(start=.45, rot=-1.75, gamma=2, hue=1.25),10)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
plt.style.use("seaborn-darkgrid")


def plot(
    function: str,
    df,
    columns="all",
    by: str = None,
    hue=None,
    ncols=2,
    plot_size=(10, 6),
    title=None,
    title_fontsize=30,
    subtitles=[],
    subtitle_fontsize=20,
    clip=None,
    kind=None,
    orient="h",
    save_path=None,
    dpi=72,
    clip_kws={},
    axis=0,
    cross_axis=0,
    bins=None,
    annot=False,
    format_num=".2f",
    ax=None,
    fig=None,
    xlabel_rotation = 45,
    **kwargs,
):
    """Function to plot multiple columns in a dataframe.

    Args:
        function (str): function name to plot.
        df (pd.DataFrame): dataframe to plot.
        columns (str, list, optional): columns to plot. Defaults to "all".
        by (str, optional): fixed column to plot by. Defaults to None.
        hue (str, optional): adds a hue semantic. Defaults to None.
        ncols (int, optional): number of columns in each row of the plot. Defaults to 2.
        plot_size (tuple, optional): Define the plot size. Defaults to (10, 6).
        title (str, optional): Superior Title. Defaults to None.
        subtitles (list, optional): Titles of each plot. If [] then it's gonna add the col's name. For not to plot subtitles use None. Defaults to [].
        clip (str, optional): clips the final plot. Options: "median" or "quantile". Defaults to None.
        kind (str, optional): Defines the kind of plot for some plots. Defaults to None.
        orient (str, optional): Defines the orientation of the plot. Defaults to "h".
        save_path (_type_, optional): Defines the path to save the plot. Defaults to None.
        dpi (int, optional): Defines the dpi of the plot. Defaults to 72.
        clip_kws (dict, optional): Defines the clip kwargs. Defaults to {}.
        axis (int, optional): Defines the axis to plot. Defaults to 0.
        cross_axis (int, optional): If cross axis is defined, it will plot the x and y axis crossed. Defaults to 0.
        bins (_type_, optional): Defines the bins for the count_lineplot. Defaults to None.
        annot (bool, optional): Defines if the plot should have annotations. Defaults to False.
        format_num (str, optional): Defines the format of the numbers for the annotations. Defaults to ".2f".
        ax (_type_, optional): Defines the axis to plot. If None it'll create an empty figure.axis. Defaults to None.
        fig (_type_, optional): Defines the figure to plot. If None it'll create an empty figure. Defaults to None.
        **kwars: key word arguments to pass to the plot function.

    Raises:
        AttributeError: _description_
    """

    if columns == "all":
        columns = df.columns
    if not isinstance(columns, list):
        columns = [columns]
    if by:
        if by in columns:
            columns = columns.drop(by)
    clip_aux = clip
    df_aux = df
    _ax = ax

    size = len(columns)
    rows = math.ceil(size / ncols)

    width, height = plot_size
    if not fig:
        fig = plt.figure(figsize=(min(ncols * width, 50), rows * height))

    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    for idx, col in enumerate(columns, start=1):
        if not _ax:
            ax = fig.add_subplot(rows, ncols, idx)

        # change axis
        if not cross_axis:
            x_aux, y_aux = col, by
        else:
            x_aux, y_aux = by, col

        if isinstance(x_aux, (list)):
            df_aux, x_aux, y_aux = df[x_aux], None, None

        # adjust limits of the analysis
        if isinstance(x_aux, str) and clip:
            clip_aux = (
                adjust_limits(df, x_aux, clip, **clip_kws)
                if orient == "h"
                else adjust_limits(df, y_aux, clip, **clip_kws)
            )

        if function == "boxplot":
            sns.boxplot(data=df_aux, x=x_aux, y=y_aux, hue=hue, ax=ax, **kwargs)

        elif function == "count_lineplot":
            _plot_count_lineplot(df=df, x=x_aux, y=y_aux, ax=ax, bins=bins, **kwargs)

        elif function == "bar_lineplot":
            _plot_bar_lineplot(df=df_aux, x=x_aux, y=y_aux, ax=ax, **kwargs)

        elif function == "plot_percentage_lineplot":
            ax = plot_percentage_lineplot(
                df=df_aux, x=x_aux, y=y_aux, hue=hue, ax=ax, **kwargs
            )

        elif function == "barplot":
            sns.barplot(data=df_aux, x=x_aux, y=y_aux, hue=hue, ax=ax, **kwargs)

        elif function == "lineplot":
            sns.lineplot(data=df_aux, x=x_aux, y=y_aux, hue=hue, ax=ax, **kwargs)

        elif function == "violinplot":
            sns.violinplot(
                data=df_aux, x=x_aux, y=y_aux, ax=ax, hue=hue, clip=clip_aux, **kwargs
            )

        elif function == "histplot":
            sns.histplot(
                data=df_aux,
                x=x_aux,
                ax=ax,
                hue=y_aux,
                binrange=clip_aux,
                kde_kws={"clip": clip_aux},
                **kwargs,
            )  # , fill=True, kde=True
            if annot:
                annotate(ax, format_num)

        elif function == "kdeplot":
            sns.kdeplot(
                data=df_aux,
                x=x_aux,
                ax=ax,
                hue=y_aux,
                fill=True,
                clip=clip_aux,
                **kwargs,
            )

        elif function == "scatterplot":
            sns.scatterplot(data=df, x=col, y=by, hue=hue, ax=ax, **kwargs)
            if clip:
                ax.set(ylim=adjust_limits(df, by, clip))

        elif function == "compare":  # columns must be a list of dicts
            if not kind:
                kind = "histplot"
            _compare_plots(
                kind=kind, df=df, col=col, by=y_aux, clip=clip, ax=ax, clip_kws=clip_kws
            )

        elif function == "prediction_error":  # columns must be a list of dicts
            if not kind:
                kind = "kde"
            _predictionerror(
                df=df, col=col, clip=clip, ax=ax, kind=kind, clip_kws=clip_kws, **kwargs
            )

        elif function == "residuals":  # columns must be a list of dicts
            _residuals(
                df=df, col=col, clip=clip, ax=ax, kind=kind, clip_kws=clip_kws, **kwargs
            )

        elif function == "density":
            _density(
                df=df,
                x=x_aux,
                y=y_aux,
                hue=hue,
                kind=kind,
                ax=ax,
                clip=clip,
                clip_kws=clip_kws,
                **kwargs,
            )

        elif function == "countplot":
            _plot_countplot(df=df, col=col, by=by, ax=ax, **kwargs)
        else:
            raise AttributeError("Not supported chart")

        if (isinstance(x_aux, str) or x_aux is None) and clip:
            if not axis:
                # ax.set_xlim(xmin=ax_min, xmax=ax_max)
                ax.set(xlim=clip_aux)
            else:
                # ax.set_ylim(ymin=ax_min, ymax=ax_max)
                ax.set(ylim=clip_aux)

        plt.xticks(rotation=xlabel_rotation, fontsize=12)
        # ax.tick_params(axis='x', rotation=45, ha='left')
        # ax.set_xticklabels(ax.get_xticklabels(),rotation=45, rotation_mode="anchor", ha='right')

        if subtitles is not None:
            subtitle = col if not subtitles else subtitles[idx - 1]
            plt.title(subtitle, fontsize=subtitle_fontsize)

    if title:
        fig.suptitle(title, fontsize=title_fontsize)
        if rows >= 2:
            fig.tight_layout()
            fig.subplots_adjust(top=0.9)

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=dpi)
    return fig, ax


def _compare_plots(kind, kde_kwargs={}, line_kwargs={}, **kwargs):
    """[Auxiliar function to help with the comparison plot]

    Args:
        kind ([type]): [histplot or kdeplot]
    """
    if not isinstance(kwargs["col"], dict):
        raise AttributeError("columns must be a list of dict")
    check_dict_keys(dict=kwargs["col"], keys=["name", "columns"])
    if not isinstance(kwargs["col"]["columns"], (list, tuple)):
        raise AttributeError("columns must be a list or a tuple'")

    if "clip" in kwargs["col"]:
        if "clip_kws" not in kwargs["col"]:
            kwargs["col"]["clip_kws"] = {}
        kwargs["clip"] = [
            adjust_limits(
                kwargs["df"],
                dist,
                clip=kwargs["col"]["clip"],
                **kwargs["col"]["clip_kws"],
            )
            for dist in kwargs["col"]["columns"]
        ]
        kwargs["clip"] = (
            min(kwargs["clip"], key=itemgetter(0))[0],
            max(kwargs["clip"], key=itemgetter(1))[1],
        )

    # looping in the columns to compare
    for i, dist in enumerate(kwargs["col"]["columns"]):
        if kind == "kdeplot":
            sns.kdeplot(
                kwargs["df"][dist],
                ax=kwargs["ax"],
                label=dist,
                clip=kwargs["clip"],
                fill=True,
                legend=True,
                alpha=0.7,
                **kde_kwargs,
            )
        elif kind == "lineplot":
            sns.lineplot(
                data=kwargs["df"],
                y=dist,
                x=kwargs["by"],
                ax=kwargs["ax"],
                color=sns.palettes.color_palette()[i],
                label=dist,
                **line_kwargs,
            )
        elif kind == "histplot":
            limits = [
                get_limits(kwargs["df"], dist, clip=kwargs["clip"])
                for dist in kwargs["col"]["columns"]
            ]
            limits = (
                min(limits, key=itemgetter(0))[0],
                max(limits, key=itemgetter(1))[1],
            )
            binwidth = (limits[1] - limits[0]) / 100
            sns.histplot(
                kwargs["df"][dist],
                # ax=kwargs["ax"],
                fill=True,
                legend=True,
                label=dist,
                binwidth=binwidth,
                binrange=kwargs["clip"],
                color=sns.palettes.color_palette()[i],
                kde=True,
                kde_kws={"clip": kwargs["clip"]},
            )
        else:
            raise NotImplementedError()
    kwargs["ax"].set(xlabel=kwargs["col"]["name"])
    kwargs["ax"].legend(loc="best")


def _predictionerror(
    kde_kws={
        "cmap": "YlOrBr",
        "kde_kws": {"color": "YlOrBr", "lw": 0.7},
        "thresh": 0.02,
        "alpha": 0.7,
    },
    line_kws={"color": "black", "linestyle": "--", "linewidth": 0.85},
    scatter_kws={"color": "black"},
    **kwargs,
):
    """[Auxiliar function to help with the prediction error plot]

    Args:
        kde_kws (dict, optional): Key word arguments for the kde. Defaults to { "cmap": "YlOrBr", "kde_kws": {"color": "YlOrBr", "lw": 0.7}, "thresh": 0.02, "alpha": 0.7, }.
        line_kws (dict, optional): Key word arguments for the line. Defaults to {"color": "black", "linestyle": "--", "linewidth": 0.85}.
        scatter_kws (dict, optional): Key word arguments for the scatter plot. Defaults to {"color": "black"}.

    Raises:
        AttributeError: "columns must be a list of dict"
    """
    dict_ = kwargs["col"]

    if not isinstance(dict_, dict):
        raise AttributeError("columns must be a list of dict")
    check_dict_keys(dict_, keys=["name", "real", "estimated"])

    if "clip" in dict_:
        kwargs["clip"] = dict_["clip"]

    r2 = r2_score(kwargs["df"][dict_["real"]], kwargs["df"][dict_["estimated"]])
    RSME = mean_squared_error(
        kwargs["df"][dict_["real"]], kwargs["df"][dict_["estimated"]], squared=False
    )
    MAE = mean_absolute_error(
        kwargs["df"][dict_["real"]], kwargs["df"][dict_["estimated"]]
    )
    MAPE = mean_absolute_percentage_error(
        kwargs["df"][dict_["real"]], kwargs["df"][dict_["estimated"]]
    )

    name = f"{dict_['name']} - R²: {r2:.3f} - RSME: {RSME:,.2f} - MAE: {MAE:,.2f} - MAPE: {MAPE:,.2%}"

    _density(
        df=kwargs["df"],
        x=dict_["real"],
        y=dict_["estimated"],
        kind=kwargs["kind"],
        ax=kwargs["ax"],
        name=name,
        draw_identity=True,
        clip=kwargs["clip"],
        squared=True,
        clip_kws=kwargs["clip_kws"],
        kde_kws=kde_kws,
        line_kws=line_kws,
        scatter_kws=scatter_kws,
    )


def _residuals(
    kde_kws={
        "cmap": "YlOrBr",
        "YlOrBr": {"color": "YlOrBr", "lw": 0.7},
        "thresh": 0.02,
        "alpha": 0.7,
    },
    line_kws={"color": "black", "linestyle": "--", "linewidth": 0.85},
    scatter_kws={"color": "black"},
    **kwargs,
):
    """_summary_

    Args:
        kde_kws (dict, optional): Key word arguments for the kde. Defaults to { "cmap": "Blues", "YlOrBr": {"color": "YlOrBr", "lw": 0.7}, "thresh": 0.02, "alpha": 0.7, }.
        line_kws (dict, optional): Key word arguments for the line. Defaults to {"color": "black", "linestyle": "--", "linewidth": 0.85}.
        scatter_kws (dict, optional): Key word arguments for the scatter points. Defaults to {"color": "black"}.

    Raises:
        AttributeError: "columns must be a list of dict"
    """
    dict_ = kwargs["col"]

    if not isinstance(dict_, dict):
        raise AttributeError("columns must be a list of dict")
    check_dict_keys(dict_, keys=["name", "estimated", "residuals"])

    if "clip" in dict_:
        kwargs["clip"] = dict_["clip"]

    _density(
        df=kwargs["df"],
        x=dict_["estimated"],
        y=dict_["residuals"],
        kind=kwargs["kind"],
        ax=kwargs["ax"],
        name=dict_["name"],
        draw_identity=False,
        clip=kwargs["clip"],
        squared=False,
        clip_kws=kwargs["clip_kws"],
        kde_kws=kde_kws,
        line_kws=line_kws,
        scatter_kws=scatter_kws,
    )


def _plot_count_lineplot(df, x, y, ax=None, bins=None, **kwargs):
    """Plots a line plot with the count of the values in the bins.

    Args:
        df (pandas.DataFrame): DataFrame with the data.
        x (str): Column name of the x axis.
        y (str): Column name of the y axis.
        ax (matplotlib.axes.Axes, optional): Axes to plot the figure. Defaults to None.
        bins (int, optional): Number of bins. Defaults to None.
    """

    if bins is not None:
        df[x] = pd.cut(df[x], bins)

    line_kwargs = {
        key.replace("line_", ""): value
        for key, value in kwargs.items()
        if "line_" in key
    }
    bar_kwargs = {
        key.replace("bar_", ""): value for key, value in kwargs.items() if "bar_" in key
    }

    sns.pointplot(data=df, x=x, y=y, ax=ax, **line_kwargs)
    ax2 = ax.twinx()
    sns.countplot(data=df, x=x, ax=ax2, **bar_kwargs)

    ax.set(ylim=(0, ax.get_yticks()[-1]))
    ax2.set(ylim=(0, ax2.get_yticks()[-1]))

    ax.set_yticks(
        np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks()))
    )
    ax2.set_yticks(
        np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
    )

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right"
    )

    # ax.xticks(rotation=45)


def _plot_bar_lineplot(df, x, y, ax=None, **kwargs):
    """Plots a line plot with a bar plot.

    Args:
        df (pandas.DataFrame): DataFrame with the data.
        x (str): Column name of the x axis.
        y (str): Column name of the y axis.
        ax (matplotlib.axes.Axes, optional): Axes to plot the figure. Defaults to None.
    """
    check_dict_keys(kwargs, ["line_y"])
    line_kwargs = {
        key.replace("line_", ""): value
        for key, value in kwargs.items()
        if "line_" in key
    }
    bar_kwargs = {
        key.replace("bar_", ""): value for key, value in kwargs.items() if "bar_" in key
    }

    sns.pointplot(data=df, x=x, ax=ax, **line_kwargs)
    ax2 = ax.twinx()
    sns.barplot(data=df, x=x, y=y, ax=ax2, **bar_kwargs)

    ax.set(ylim=(0, ax.get_yticks()[-1]))
    ax2.set(ylim=(0, ax2.get_yticks()[-1]))

    ax.set_yticks(
        np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks()))
    )
    ax2.set_yticks(
        np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
    )

    ax.set_xticklabels(
        ax.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right"
    )


def _plot_countplot(
    df, col, by, ax, rotation=45, ha="center", annot_fontsize=12, **kwargs
):
    """[plot a countplot]

    Args:
        df ([pd.DataFrame]): [pandas Dataframe]
        col ([List]): [Column to make the plot].
        by ([str]): [Target columname].
        ax ([plt.ax]: [Axes object].
    """

    if by:
        ylim = (
            df[[col, by]]
            .groupby([col, by])
            .agg({by: "count"})
            .nlargest(1, by)[by]
            .values
            * 1.2
        )
        ax.set(ylim=(0, ylim))
        
        if 'order' in kwargs:
            order = kwargs.pop('order')
        else:
            order = df[col].value_counts().index 
            
        chart = sns.countplot(
            x=df[col], hue=df[by], ax=ax, order=order **kwargs
        )
        n_categoria = -1
        try:
            total = len(df[by])
            for idx, p in enumerate(ax.patches):
                if idx % len(df[col].unique()) == 0:
                    n_categoria = -1
                n_categoria += 1
                categoria = chart.get_xticklabels()[n_categoria].get_text()

                total = len(df[df[col].astype(str) == categoria])

                annot = f"{p.get_height():,.0f} - {(p.get_height()/total)*100:.2f}%"
                ax.annotate(
                    annot,
                    (p.get_x() + p.get_width() / 2, p.get_height() + ylim * 0.02),
                    fontsize=annot_fontsize,
                    rotation=rotation,
                    ha=ha,
                )
        except:
            print(f"{idx} - {p} - {n_categoria}")

    else:
        ylim = (
            df[[col]].groupby([col]).agg({col: "count"}).nlargest(1, col)[col].values
            * 1.2
        )
        ax.set(ylim=(0, ylim))
        
        if 'order' in kwargs:
            order = kwargs.pop('order')
        else:
            order = df[col].value_counts().index 
            
        chart = sns.countplot(x=df[col], ax=ax, order=order, **kwargs)
        n_categoria = -1
        total = len(df[col])
        for p in ax.patches:
            annot = f"{p.get_height():,.0f} - {p.get_height()/total*100:.2f}%"
            ax.annotate(
                annot,
                (p.get_x() + p.get_width() / 2, p.get_height() + ylim * 0.02),
                fontsize=annot_fontsize,
                rotation=rotation,
                ha=ha,
            )


def _density(
    df,
    x,
    y,
    kind=None,
    ax=None,
    hue=None,
    name=None,
    draw_identity=False,
    clip="quantile",
    squared=False,
    best_fit=True,
    clip_kws={},
    kde_kws={},
    line_kws={},
    scatter_kws={},
    **kwargs,
):
    """Plots a density plot.

    Args:
        df (pandas.DataFrame): DataFrame with the data.
        x (str): Column name of the x axis.
        y (str): Column name of the y axis.
        kind (str, optional): Kind of plot. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to plot the figure. Defaults to None.
        hue (str, optional): Column name to group the data. Defaults to None.
        name (str, optional): Name of the plot. Defaults to None.
        draw_identity (bool, optional): Draw the identity line. Defaults to False.
        clip (str, optional): Method to clip the plot. Defaults to "quantile".
        squared (bool, optional): If True, the plot will be squared. Defaults to False.
        best_fit (bool, optional): If True, the best fit line will be drawn. Defaults to True.
        clip_kws (dict, optional): Keyword arguments for the clip method. Defaults to {}.
        kde_kws (dict, optional): Keyword arguments for the kde method. Defaults to {}.
        line_kws (dict, optional): Keyword arguments for the line method. Defaults to {}.
        scatter_kws (dict, optional): Keyword arguments for the scatter method. Defaults to {}.
    """
    # get coefs of linear fit
    if best_fit:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x], df[y])
        label_line = f"Best Fit - y={slope:.3f}x {intercept:+.3f} - R²: {r_value:.3f}"
        sns.regplot(
            data=df,
            x=x,
            y=y,
            scatter=kind == "scatter",
            label=label_line,
            ax=ax,
            line_kws=line_kws,
            scatter_kws=scatter_kws,
        )

    label = name if name else x if not hue else None

    try:
        sns.kdeplot(data=df, x=x, y=y, hue=hue, ax=ax, label=label, **kde_kws)
        sns.kdeplot(data=df, x=x, y=y, hue=hue, fill=True, ax=ax, cbar=False, **kde_kws)
    except:
        pass

    if draw_identity:
        draw_identity_line(ax=ax, label="Identity line")

    ax.legend(loc="upper left")

    if clip:
        ax.set(xlim=adjust_limits(df, x, clip=clip, **clip_kws))
        if squared:
            ax.set(ylim=adjust_limits(df, x, clip=clip, **clip_kws))
        else:
            ax.set(ylim=adjust_limits(df, y, clip=clip, **clip_kws))


# def plot_runtimeline(df, runtime_col, cols, by, title='Features x Runtime'):
#     """[Return the plot of the variables in relation to the runtime]

#     Args:
#         df ([df]): [pandas or koalas dataframe]
#         runtime_col ([str]): [string with runtime columname]
#         drop_col ([list]): [list with coluns to drop]
#         id_col ([str]): [string with runtime columname, only necessary when method == 'single']
#         single_id ([str,int]): [id to be selected, only necessary when method == 'single']
#         method (str, optional): [Option of method, 'mean' or 'single']. Defaults to 'mean'.
#         title ([str], optional): [Title of the plot]. Defaults to None.
#     """
#     # if method =='mean':
#     #     df_final = df.drop(drop_col, axis=1).groupby(runtime_col).mean()
#     # elif method=='single':
#     #     df_final = df[df[id_col] == single_id]

#     rows = len(cols)
#     grouped = df.groupby(by)
#     for key, group in grouped:
#         group.plot(subplots=True,x=runtime_col,y=cols,sharex=True,figsize=(20,rows*5), title = title)


def heat_map_corr(df, columns=None, title=None, figsize=None, save_path=None, dpi=72):
    """[Return the plot of the variable Pearson's Correlation]

    Args:
        df ([df]): [pandas or koalas dataframe]
        columns ([List], optional): [Columns to iterate and make the suplots]. Defaults to None.
        title ([str], optional): [Title of the plot]. Defaults to None.
    """
    if columns == None:
        columns = df.columns
    # Compute the correlation matrix
    corr = df[columns].corr()
    size = len(columns)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    figsize = figsize if figsize else (size * 1.5, size * 1.5)
    f, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.7,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=True,
        annot_kws={"size": 12},
        fmt=".2f",
    )

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        f.savefig(save_path, dpi=dpi)


def compare_model_results(
    df, names, metrics, hue=None, save_path=None, plot_size=(3, 6), dpi=72
):
    """[Plot the model comparison]

    Args:
        df ([pd.DataFrame]): [DataFrame with the results of the models]
        names ([str]): [String for the column with the models names ]
        metrics ([list]): [List with the columnames for the metrics]
        hue ([str], optional): [Column name to group the data]. Defaults to None.
        save_path ([str], optional): [Path to save the plot]. Defaults to None.
        plot_size ([tuple], optional): [Size of the plot]. Defaults to (3, 6).
        dpi (int, optional): [Dpi of the plot]. Defaults to 72.
    """
    # boxplot algorithm comparison
    sns.set(font_scale=0.9)
    if not iter(metrics):
        metrics = [metrics]

    rows = len(metrics)
    cols = df[names].nunique()

    width, height = plot_size

    fig = plt.figure(figsize=(min(cols * width, 50), height * rows))
    fig.suptitle("Models' Comparison", fontsize=22)
    fig.subplots_adjust(hspace=0.8, wspace=0.2)

    for idx, metric in enumerate(metrics, start=1):
        ax = fig.add_subplot(rows, 1, idx)
        sns.boxplot(data=df, x=names, y=metric, hue=hue)
        ax.set_xticklabels(
            list(df[names].unique()),
            horizontalalignment="right",
            fontsize="x-large",
            rotation=45,
        )
        ax.set_ylabel(metric, fontsize=24)
        ax.set_xlabel("", fontsize=20)

    if rows >= 2:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=dpi)

    # plt.show()


def plot_percentage_lineplot(df, x, y, hue, ax=None, bins=None, **kwargs):
    """Plot a stacked barplot with a line plot over it.

    Args:
        df ([pd.DataFrame]): [DataFrame with the data]
        x ([str]): [Column name for the x axis]
        y ([str]): [Column name for the y axis]
        hue ([str]): [Column name for the hue]
        ax ([matplotlib.axes.Axes], optional): [Axes to plot the plot]. Defaults to None.
        bins ([int], optional): [Number of bins for the histogram]. Defaults to None.

    Returns:
        [matplotlib.axes.Axes]: [Axes with the plot]
    """

    if bins is not None:
        df[x] = pd.cut(df[x], bins)

    line_kwargs = {
        key.replace("line_", ""): value
        for key, value in kwargs.items()
        if "line_" in key
    }
    bar_kwargs = {
        key.replace("bar_", ""): value for key, value in kwargs.items() if "bar_" in key
    }

    sns.histplot(data=df, x=x, ax=ax, hue=hue, **bar_kwargs)
    ax2 = ax.twinx()
    sns.pointplot(data=df, x=x, ax=ax2, y=y, **line_kwargs)

    ax.set(ylim=(0, ax.get_yticks()[-1]))
    ax2.set(ylim=(0, ax2.get_yticks()[-1]))

    ax.set_yticks(
        np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], len(ax.get_yticks()))
    )
    ax2.set_yticks(
        np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax.get_yticks()))
    )

    ax.set_xticklabels(
        ax2.get_xticklabels(), rotation=45, rotation_mode="anchor", ha="right"
    )

    return ax


def plot_residuals(
    model,
    X_test,
    y_test,
    X_train=None,
    y_train=None,
    is_fitted=True,
    cmap="RdBu",
    ax=None,
):
    """[Plot Residuals]

    Args:
        model ([model]): [Machine Learning model]
        X_test ([pd.DataFrame]): [The data to predict]
        y_test ([pd.Series]): [The target variable with the value to be predicted]
        X_train ([pd.DataFrame]): [The data to fit]
        y_train ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        is_fitted (bool, optional): [if the model is fitted or not]. Defaults to True.
        cmap (str, optional): [color map]. Defaults to 'RdBu'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    visualizer = ResidualsPlot(model, cmap=cmap, ax=ax)
    if not is_fitted:
        visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    sns.set(font_scale=1.4)
    visualizer.show()  # Finalize and show the figure


def plot_predictionerror(
    model,
    X_test,
    y_test,
    X_train=None,
    y_train=None,
    is_fitted=True,
    cmap="RdBu",
    ax=None,
):
    """[Plot Prediction Error]

    Args:
        model ([model]): [Machine Learning model]
        X_test ([pd.DataFrame]): [The data to predict]
        y_test ([pd.Series]): [The target variable with the value to be predicted]
        X_train ([pd.DataFrame]): [The data to fit]
        y_train ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        is_fitted (bool, optional): [if the model is fitted or not]. Defaults to True.
        cmap (str, optional): [color map]. Defaults to 'RdBu'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    visualizer = PredictionError(model, cmap=cmap, ax=ax)
    if not is_fitted:
        visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    sns.set(font_scale=1.4)
    visualizer.show()  # Finalize and show the figure


def plot_predictionerror_train_test(
    model, X_train, y_train, X_test, y_test, is_fitted=True, cmap="RdBu"
):
    """[Plot Prediction Error for train and test]

    Args:
        model ([model]): [Machine Learning model]
        X_train ([pd.DataFrame]): [The data to fit]
        y_train ([pd.Series]): [The target variable to try to predict in the case of supervised learning]
        X_test ([pd.DataFrame]): [The data to predict]
        y_test ([pd.Series]): [The target variable with the value to be predicted]
        is_fitted (bool, optional): [if the model is fitted or not]. Defaults to True.
        cmap (str, optional): [color map]. Defaults to 'RdBu'.
    """
    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 2, 1)
    visualizer = PredictionError(model, cmap=cmap, ax=ax)
    if not is_fitted:
        visualizer.fit(X_train, y_train)
    visualizer.score(X_train, y_train)
    visualizer.finalize()
    ax.set_title("Train - PredictionError", fontsize=20)

    ax = fig.add_subplot(1, 2, 2)
    visualizer = PredictionError(model, support=True, cmap=cmap, ax=ax)
    if not is_fitted:
        visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.finalize()
    ax.set_title("Test - PredictionError", fontsize=20)


def plot_model_history(history, ncols=2, save_path=None, plot_size=(10, 6), dpi=72):
    """[Plot Model History]

    Args:
        history ([type]): [Tensorflow Model History]
        ncols (int, optional): [Number of columns for each row]. Defaults to 2.
        save_path ([type], optional): [Save path]. Defaults to None.
        plot_size (tuple, optional): [Plot size]. Defaults to (10, 6).
        dpi (int, optional): [DPI]. Defaults to 72.
    """
    metrics = [metric for metric in history.keys() if "val_" not in metric]

    size = len(metrics)
    rows = math.ceil(size / 2)

    width, height = plot_size

    fig = plt.figure(figsize=(min(ncols * width, 50), rows * height))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    for idx, metric in enumerate(metrics, start=1):

        ax = fig.add_subplot(rows, ncols, idx)

        sns.lineplot(y=history[metric], x=range(1, len(history[metric]) + 1), ax=ax)
        if not metric == "lr" and "val_" + str(metric) in history.keys():
            sns.lineplot(
                y=history["val_" + str(metric)],
                x=range(1, len(history["val_" + str(metric)]) + 1),
                ax=ax,
            )

        ax.legend(["train", "validation"], loc="upper left")

        ax.set(title=(f"{metric}"), ylabel=(f"{metric}"), xlabel=("epoch"))

    if rows >= 2:
        fig.tight_layout()
        fig.subplots_adjust(top=0.95)

    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        fig.savefig(save_path, dpi=dpi)


def plot_confusion_matrix_train_test(y_train, y_train_pred, y_test, y_pred, **kwargs):
    """[Plot Confusion Matrix for train and test results]

    Args:
        y_train ([type]): [[The target variable to try to predict in the case of supervised learning]]
        y_train_pred ([type]): [The target variable with the predicted value for the train]
        y_test ([type]): [The target variable with the real value for the test]
        y_pred ([type]): [The target variable with the predicted value for the test]
    """

    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(hspace=0.8, wspace=0.2)
    ax = fig.add_subplot(1, 2, 1)
    plot_confusion_matrix(
        y_train, y_train_pred, title="Train - Confusion Matrix", ax=ax, **kwargs
    )
    ax = fig.add_subplot(1, 2, 2)
    plot_confusion_matrix(
        y_test, y_pred, title="Test - Confusion Matrix", ax=ax, **kwargs
    )


def plot_confusion_matrix(
    y_real, y_pred, title="Confusion Matrix", ax=None, figsize=(10, 6), annot_size=20
):
    """[Plot Confusion Matrix]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Confusion Matrix'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    cf_matrix = confusion_matrix(y_real, y_pred)
    if not ax:
        f, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20)

    n_categories = len(np.unique(y_real))

    group_names = (
        ["True Neg\n", "False Pos\n", "False Neg\n", "True Pos\n"]
        if not cf_matrix.shape[0] > 2
        else ["" for i in range(len(cf_matrix.flatten()))]
    )
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [
        "{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)
    ]
    labels = [
        f"{v1}{v2}\n({v3})"
        for v1, v2, v3 in zip(group_names, group_counts, group_percentages)
    ]
    labels = np.asarray(labels).reshape(n_categories, n_categories)
    sns.heatmap(
        cf_matrix,
        annot=labels,
        fmt="",
        cmap="RdBu",
        ax=ax,
        annot_kws={"size": annot_size},
    )
    sns.set(font_scale=1.4)
    ax.set_ylabel("True Label", fontsize=18)
    ax.set_xlabel("Predicted Label", fontsize=18)


def plot_classification_report_train_test_sns(
    y_train, y_train_pred, y_test, y_pred, figsize=(20, 6), **kwargs
):
    """[Plot Confusion Matrix for train and test results]

    Args:
        y_train ([type]): [[The target variable to try to predict in the case of supervised learning]]
        y_train_pred ([type]): [The target variable with the predicted value for the train]
        y_test ([type]): [The target variable with the real value for the test]
        y_pred ([type]): [The target variable with the predicted value for the test]
    """

    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(hspace=0.8, wspace=0.2)
    ax = fig.add_subplot(1, 2, 1)
    plot_classification_report_sns(
        y_train, y_train_pred, title="Train - Classification Report", ax=ax, **kwargs
    )
    ax = fig.add_subplot(1, 2, 2)
    plot_classification_report_sns(
        y_test, y_pred, title="Test - Classification Report", ax=ax, **kwargs
    )


def plot_classification_report_sns(
    y_real,
    y_pred,
    title="Classification Report",
    ax=None,
    figsize=(10, 6),
    annot_size=20,
):
    """[Plot Classification Report]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Classification Report'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    df = pd.DataFrame(
        classification_report(y_real, y_pred, digits=2, output_dict=True)
    ).T
    df["support"] = df.support.apply(int)
    mask = np.zeros(df.shape)
    mask[:, -1] = True

    if not ax:
        f, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=20)
    sns.heatmap(
        df,
        cmap="RdBu",
        annot=True,
        fmt=".2f",
        ax=ax,
        annot_kws={"size": annot_size},
        vmin=0,
        vmax=1,
        mask=mask,
    )
    for (j, i), label in np.ndenumerate(df.values):
        if i == df.shape[1] - 1:
            ax.text(
                i + 0.5,
                j + 0.5,
                label,
                fontdict=dict(
                    ha="center", va="center", color="black", fontsize=annot_size
                ),
            )

    sns.set(font_scale=1.4)


def plot_roc_train_test_sns(
    y_train, y_train_pred, y_test, y_pred, labels=["Class 1", "Class 2"], **kwargs
):
    """[Plot Roc Curve for train and test results]

    Args:
        y_train ([type]): [[The target variable to try to predict in the case of supervised learning]]
        y_train_pred ([type]): [The target variable with the predicted value for the train]
        y_test ([type]): [The target variable with the real value for the test]
        y_pred ([type]): [The target variable with the predicted value for the test]
    """

    fig = plt.figure(figsize=(20, 6))
    fig.subplots_adjust(hspace=0.8, wspace=0.2)
    ax = fig.add_subplot(1, 2, 1)
    plot_roc_sns(
        y_train,
        y_train_pred,
        labels=labels,
        title="Train - Classification Report",
        ax=ax,
        **kwargs,
    )
    ax = fig.add_subplot(1, 2, 2)
    plot_roc_sns(
        y_test,
        y_pred,
        labels=labels,
        title="Test - Classification Report",
        ax=ax,
        **kwargs,
    )


def plot_roc_sns(
    y_real, y_pred, title="ROC Curve", labels=["Class 1", "Class 2"], ax=None, **kwargs
):
    """[Plot ROC Curve]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'ROC Curve'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    # Visualisation with plot_metric
    bc = BinaryClassification(y_real, y_pred, labels=labels, **kwargs)
    # Figures
    bc.plot_roc_curve()
    if title:
        ax.set_title(title, fontsize=20)


def plot_precision_recall_vs_threshold(
    y_real, y_pred, title="Recall and Precision x Threshold Curve", ax=None
):
    """[Plot Precision and Recall vs Threshold Curve]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Recall and Precision x Threshold Curve'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    if title:
        ax.set_title(title, fontsize=20)

    precisions, recalls, thresholds = precision_recall_curve(y_real, y_pred)

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")


def plot_precision_vs_recall(
    y_real, y_pred, title="Recall x Precision", ax=None, **kwargs
):
    """[Plot Recall x Precision]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Recall x Precision'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    # Visualisation with plot_metric
    bc = BinaryClassification(y_real, y_pred, labels=["Class 1", "Class 2"], **kwargs)
    # Figures
    bc.plot_precision_recall_curve()
    if title:
        ax.set_title(title, fontsize=20)


def plot_class_distribution(
    y_real, y_pred, title="Class Distribution", ax=None, **kwargs
):
    """[Plot Class Distribution]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Class Distribution'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    # Visualisation with plot_metric
    bc = BinaryClassification(y_real, y_pred, labels=["Class 1", "Class 2"], **kwargs)
    # Figures
    bc.plot_class_distribution()
    if title:
        ax.set_title(title, fontsize=20)


def plot_DET_curve(
    y_real, y_pred, title="Detection error tradeoff (DET)", ax=None, **kwargs
):
    """[Plot Detection error tradeoff (DET)]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'Detection error tradeoff (DET)'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    if title:
        ax.set_title(title, fontsize=20)

    DetCurveDisplay.from_predictions(y_real, y_pred, ax=ax, **kwargs)


def plot_fpr_fnr_vs_threshold(
    y_real, y_pred, title="FPR and FNR x Threshold Curve", ax=None, **kwargs
):
    """[Plot FPR and FNR x Threshold Curve]

    Args:
        y_real ([type]): [The target variable with the real value]
        y_pred ([type]): [The target variable with the predicted value]
        title (str, optional): [Title of the plot]. Defaults to 'FPR and FNR x Threshold Curve'.
        ax ([type], optional): [Axes object]. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    if title:
        ax.set_title(title, fontsize=20)

    fpr, fnr, thresholds = det_curve(y_real, y_pred)

    plt.plot(thresholds, fpr, "b--", label="False Positive Rate (FPR)")
    plt.plot(thresholds, fnr, "g-", label="False Negative Rate (FNR)")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc="best")


def plot_calibration_curve(
    y_real, y_pred, title="Calibration Curve", ax=None, **kwargs
):
    """Plot Calibration Curve

    Args:
        y_real (np.array, pd.series): The target variable with the real value
        y_pred (np.array, pd.series): The target variable with the predicted probability value
        title (str, optional): Title. Defaults to 'Calibration Curve'.
        ax (ax, optional): Matplotlib ax. Defaults to None.
    """
    if not ax:
        f, ax = plt.subplots(figsize=(10, 6))

    if title:
        ax.set_title(title, fontsize=20)

    CalibrationDisplay.from_predictions(y_real, y_pred, ax=ax, **kwargs)
