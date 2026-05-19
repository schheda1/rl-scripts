import argparse

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_apps():
    # returns a list of all applications
    apps = []
    apps.append("bezier-surface")
    apps.append("mandelbrot")
    apps.append("bspline-vgh")
    apps.append("contract")
    apps.append("haccmk")
    apps.append("ccs")
    apps.append("coordinates")
    apps.append("xsbench")
    apps.append("rainflow")
    apps.append("bn")
    apps.append("complex")
    apps.append("lavamd")
    apps.append("quicksort")
    apps.append("libor")
    apps.append("clink")
    apps.append("qtclustering")
    return apps


def app_pretty(s):
    # returns a pretty version of the application name
    app_2_pretty = {}
    app_2_pretty["atomicAggregate"] = "atomic\naggregate"
    app_2_pretty["bezier-surface"] = "bezier\nsurface"
    app_2_pretty["bspline-vgh"] = "bspline\nvgh"
    app_2_pretty["coordinates"] = "coor-\ndinates"
    app_2_pretty["contract"] = "contract"
    app_2_pretty["mandelbrot"] = "mandel-\nbrot"
    app_2_pretty["rainflow"] = "rainflow"
    app_2_pretty["xsbench"] = "XSBench"
    app_2_pretty["lavamd"] = "lavaMD"
    app_2_pretty["quicksort"] = "quicksort"
    app_2_pretty["qtclustering"] = "qtclus-\ntering"
    if s in app_2_pretty:
        return app_2_pretty[s]

    # add '-' and line break after every 7th character
    result = ""
    for i, c in enumerate(s):
        result += c
        if (i + 1) % 7 == 0 and (i + 1) != len(s):
            result += "-\n"
    return result


def datadump_pretty(datadump):
    # turns e.g.: unroll7/data_dump_std_unroll_7.csv into unroll 7
    datadump = datadump.split("/")[-1]
    split_undescore = datadump.split("_")
    config = split_undescore[-2]
    unrolling = str(int(split_undescore[-1][:-4]))

    if "heuristic" in config:
        return "heuristic"
    if config == "uu":
        return "u&u " + unrolling
    if config == "unmerge":
        return "unmerge"
    return config + " " + unrolling


def get_app_name(app):
    if "/" in app:
        app = app.split("/")[-1]
    return app


def get_apps_and_rows(datadumps, columns):
    datadump_2_df = {}

    failed_datadumps = []
    for datadump in datadumps:
        print("Reading: ", datadump)
        try:
            df = pd.read_csv(datadump, sep=";", names=columns)
        except Exception as e:
            print("WARNING: could not read: ", datadump)
            failed_datadumps.append(datadump)
            continue
        datadump_2_df[datadump] = df

    for failed_datadump in failed_datadumps:
        datadumps.remove(failed_datadump)

    apps = get_apps()
    apps.sort()
    default_2_comp_time = {}
    rows = []

    for datadump, df in datadump_2_df.items():
        for index, row in df.iterrows():
            application_name = get_app_name(row["app"])
            if not application_name in apps:
                continue
            loop = row["loop"]
            if loop == "default":
                default_2_comp_time[application_name] = row["comp_time"]

    for datadump, df in datadump_2_df.items():
        for index, row in df.iterrows():
            application_name = get_app_name(row["app"])
            if not application_name in apps:
                continue
            loop = row["loop"]
            if loop == "default":
                continue
            speedup = row["speedup"]
            codesize = row["codeSizeIncrease"]
            compile_time = row["comp_time"] / default_2_comp_time[application_name]
            if "heuristic" in loop or not "_" in loop:
                # if _ in loop, then this entry is one where our pass was applied to multiple loops (except for heuristic)
                datadump_prettified = datadump_pretty(datadump)
                app_prettified = app_pretty(application_name)
                entry = (
                    datadump_prettified,
                    app_prettified,
                    speedup,
                    compile_time,
                    codesize,
                )
                rows.append(entry)

    return apps, rows


def set_margins(fig, margins):
    """Set figure margins as [left, right, top, bottom] in inches
    from the edges of the figure."""
    left, right, top, bottom = margins
    width, height = fig.get_size_inches()

    # convert to figure coordinates:
    left, right = left / width, 1 - right / width
    bottom, top = bottom / height, 1 - top / height

    # get the layout engine and convert to its desired format
    try:
        engine = fig.get_layout_engine()
        if isinstance(engine, matplotlib.layout_engine.TightLayoutEngine):
            rect = (left, bottom, right, top)
        elif isinstance(engine, matplotlib.layout_engine.ConstrainedLayoutEngine):
            rect = (left, bottom, right - left, top - bottom)
        else:
            raise RuntimeError("Cannot adjust margins of unsupported layout engine")

        # set and recompute the layout
        engine.set(rect=rect)
        engine.execute(fig)
    except Exception as e:
        print("Failed to set margins: {}".format(e))


def violin_plot(datadumps, y="speedup", name=None, disable_plot=False):
    if y == "comptime":
        y = "comp_time"

    columns = [
        "app",
        "loop",
        "speedup",
        "loopSize",
        "codeSizeIncrease",
        "comp_time",
        "codesize",
        "std",
    ]

    apps, rows = get_apps_and_rows(datadumps, columns)

    df = pd.DataFrame(
        rows, columns=["config", "app", "speedup", "comp_time", "codesize"]
    )

    sns.set(style="whitegrid")
    sns.set_palette("colorblind")
    plt.rcParams["font.size"] = 100
    sns.set_context("notebook", font_scale=1.2)

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    ax = []
    n_plots = 16
    for i in range(n_plots):
        ax.append(fig.add_subplot(2, 8, i + 1))

    # creates some space at the top for the legend
    set_margins(fig, [0 * 11, 0 * 11, 0.048 * 8.5, 0 * 8.5])

    i = 0
    df_heuristic = df[df["config"] == "heuristic"]
    includes_heuristic = len(df_heuristic) > 0
    df = df[df["config"] != "heuristic"]

    # Create color mapping so that each config is always mapped to the same color
    # If measurements for some applications of a config are missing, the color mapping would be wrong otherwise
    num_categories = len(datadumps)
    if includes_heuristic:
        num_categories -= 1
    colorblind_palette = sns.color_palette("colorblind", num_categories)
    category_list = df["config"].unique()
    color_mapping = dict(zip(category_list, colorblind_palette))

    for app in apps:
        app = app_pretty(app)
        df_app = df[df["app"] == app]
        if len(df_app) > 0:
            sns.boxplot(data=df_app, x="app", y=y, hue="config", ax=ax[i], boxprops=dict(alpha=0.3), showfliers=False, palette=color_mapping)
            sns.stripplot(data=df_app, x="app", y=y, hue="config", ax=ax[i], dodge=True, palette=color_mapping)
            ax[i].set_xlabel("", fontsize=100)
            ax[i].set_ylabel("", fontsize=100)
        else:
            print("WARNING: no measurements for benchmark: ", app)
            try:
                ax[i].set_xticklabels([app])
            except Exception as e:
                print("WARNING: could not set xticklabels for benchmark: ", app)


        if includes_heuristic:
            df_app_heuristic = df_heuristic[df_heuristic["app"] == app]
            values = df_app_heuristic[y].values
            if len(values) == 0:
                print("WARNING: no heuristic measurements for benchmark: ", app)
                ax[i].axhline(-0.1, ls="--", color="red", linewidth=2, label="u&u heuristic", )
            else:
                heuristic_value = df_app_heuristic[y].values[0]
                ax[i].axhline(heuristic_value, ls="--", color="red", linewidth=2, label="u&u heuristic", )

        if ax[i].get_legend() is not None:
            ax[i].get_legend().remove()
        i += 1

    new_handles, new_labels = get_handles_labels(datadumps, ax, includes_heuristic)

    leg = plt.legend(new_handles, new_labels, loc="upper center", bbox_to_anchor=(-4.05, 2.54), ncol=len(new_handles),
        fancybox=True, shadow=True, )
    leg.set_in_layout(False)
    if name is None:
        name = "boxplot-" + y
        if includes_heuristic:
            name += "-heuristic"
    plt.savefig(name + ".pdf", format="pdf")
    if not disable_plot:
        plt.show()


def get_handles_labels(datadumps, ax, includes_heuristic):
    num_datadumps = len(datadumps)
    if includes_heuristic:
        num_datadumps -= 1
    handles, labels = ax[0].get_legend_handles_labels()

    new_handles = []
    new_handles.extend(handles[:num_datadumps])
    if includes_heuristic:
        new_handles.append(handles[-1])
    new_labels = []
    new_labels.extend(labels[:num_datadumps])
    if includes_heuristic:
        new_labels.append(labels[-1])
    return new_handles, new_labels


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datadumps",
        nargs="+",
        type=str,
        help="Path to one or more files containing execution times logs.",
    )
    parser.add_argument(
        "--y",
        dest="y",
        type=str,
        help="y=comptime OR y=speedup OR y=codesize",
    )
    parser.add_argument("--name", dest="name", type=str, help="Name of the output file")
    parser.add_argument(
        "--disable-plot",
        dest="disable_plot",
        action="store_true",
        help="Disable plotting",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parseArgs()
    violin_plot(
        args.datadumps, y=args.y, name=args.name, disable_plot=args.disable_plot
    )
