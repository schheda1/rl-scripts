import argparse
import os
from collections import defaultdict
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from loop_2_kernel import *
from loop_data import *
from executors.util import *

medians = []
avgs = []
stds = []
conf_intervals = []
loop_indices = []
names = []
filenames = []
median_compilation_times = []
num_unrollings = []
num_iterations = []
codesizes = []

# maps loops to their kernels they are contained in
app_2_loop_2_kernels = {}


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('execution_times', nargs='+', type=str,
                        help='Path to one or more files containing execution times logs.')
    parser.add_argument('--metadata', type=str, help='Path to loop metadata.')
    parser.add_argument('--o', type=str, help='Output resulting dataframe.')
    parser.add_argument('--sort-by', type=str, help='Sort output by given column name.')
    parser.add_argument('--only-single-loops', dest='singleLoops', action='store_true')
    parser.add_argument('--sort-reverse', dest='sort_reverse', action='store_true')
    parser.add_argument('--precision', nargs='?', default=3, type=int)
    parser.add_argument('--full-output', dest='full_output', action='store_true')
    parser.add_argument('--compare', dest='compareSpeedup', action='store_true')
    parser.add_argument('--late', dest='late', action='store_true')
    parser.add_argument('--no-late', dest='no_late', action='store_true')
    parser.add_argument('--compact-map', dest='compact_map', action='store_true')
    parser.add_argument('--interesting', nargs='?', dest='interesting', type=str)
    parser.add_argument('--name', nargs='?', dest='name', type=str)
    parser.add_argument('--cmp-size', nargs='?', dest='cmp_size', type=str)
    parser.add_argument('--plot-O3-scatter', dest='plot_O3_scatter', type=str)
    parser.add_argument('--plot-O3-scatter-dir', dest='plot_O3_scatter_dir', type=str)
    parser.add_argument('--use-nvprof', nargs='?', dest='use_nvprof', type=str, default=None, const="ALL_KERNELS")
    parser.add_argument('--heuristic-full-name', dest='heuristicFullName', action='store_true')
    parser.add_argument('--table', dest='table', type=str)
    parser.add_argument("--disable-plot", dest="disable_plot", action="store_true", help="Disable plotting")
    parser.set_defaults(sort_by='speedup')
    parser.set_defaults(singleLoops=False)
    parser.set_defaults(full_output=False)
    parser.set_defaults(compareSpeedup=False)
    parser.set_defaults(late=False)
    parser.set_defaults(no_late=False)
    parser.set_defaults(compact_map=False)
    parser.set_defaults(heuristicFullName=False)
    args = parser.parse_args()
    return args


def create_df(args):
    # csv to df
    dfs = []
    for execution_time_log in args.execution_times:
        df = pd.read_csv(execution_time_log, sep=";")
        if args.singleLoops:
            df = df.drop(df[df.loopIdx < -2].index)
        dfs.append(df)

    # concat execution_time dfs
    df = dfs[0]
    for i in range(1, len(dfs)):
        df = pd.concat([df, dfs[i]])

    return df


def reset_globals():
    global medians, avgs, stds, conf_intervals, loop_indices, names, filenames, median_compilation_times, num_unrollings
    global num_iterations, codesizes
    medians = []
    avgs = []
    stds = []
    conf_intervals = []
    loop_indices = []
    names = []
    filenames = []
    median_compilation_times = []
    num_unrollings = []
    num_iterations = []
    codesizes = []


def create_df_compare(args):
    # csv to df
    dfs = []
    if args.late and args.no_late:
        print("ERROR: --late and --no-late are mutually exclusive")
        quit()
    if args.late:
        args.execution_times = [x for x in args.execution_times if "late" in x]
    if args.no_late:
        args.execution_times = [x for x in args.execution_times if "late" not in x]

    # remove non-existing files
    existing_files = []
    for execution_time_log in args.execution_times:
        print("Reading", execution_time_log)
        try:
            df = pd.read_csv(execution_time_log, sep=";")
            existing_files.append(execution_time_log)
        except:
            print("ERROR: could not read file", execution_time_log)
            continue

    args.execution_times = existing_files
    if len(args.execution_times) == 0:
        print("ERROR: no files to read")
        quit()
    for execution_time_log in args.execution_times:
        try:
            df = pd.read_csv(execution_time_log, sep=";")
        except:
            print("ERROR: could not read file", execution_time_log)
            continue
        if args.singleLoops:
            df = df.drop(df[df.loopIdx < -2].index)
        dfs.append(df_create_stats(df))

    # new df: each row = loop_idx, each column = speedup over default
    config_2_speedups = defaultdict(list)
    loop_2_idx = defaultdict(int)
    row_names = []
    df_idx = 0
    name_column = []
    for df in dfs:
        for index, row in df.iterrows():
            name = row['name']
            if name == 'default':
                continue
            if row['loopIdx'] < 0:
                continue
            if not name in loop_2_idx:
                rowIdx = len(loop_2_idx)
                loop_2_idx[name] = rowIdx
                name_column.append(name)
    config_2_speedups['names'] = name_column

    unrolling = max([get_num_unrolling(x) for x in args.execution_times])
    print("unrolling", unrolling)
    if args.compact_map:
        args.execution_times = [filename_2_config(x) for x in args.execution_times]

    name_2_idx = {}
    idx_2_name = {}
    for df in dfs:
        df_name = args.execution_times[df_idx]
        print(df_name, "->", df_idx)
        name_2_idx[df_name] = df_idx
        idx_2_name[df_idx] = df_name
        for index, row in df.iterrows():
            name = row['name']
            if name == 'default':
                continue
            if row['loopIdx'] < 0:
                continue
            speedup = row['speedup']
            rowIdx = loop_2_idx[name]
            speedups = config_2_speedups[df_idx]
            if len(speedups) == 0:
                speedups = [0] * len(loop_2_idx)
            speedups[rowIdx] = speedup
            config_2_speedups[df_idx] = speedups
        df_idx += 1

    result_df = pd.DataFrame(config_2_speedups)

    if args.interesting is not None:
        rows_to_delete = []
        diffs = []
        arg_maxs = []
        speedups = []
        for index, row in result_df.iterrows():
            max_force_unroll = 0
            max_unmerge = 0
            argmax_force_unroll = -1
            argmax_unmerge = -1
            for idx, name in idx_2_name.items():
                if not idx in result_df.columns:
                    continue
                speedup = row[idx]
                if "unmerge" in name:
                    if speedup > max_unmerge:
                        max_unmerge = speedup
                        argmax_unmerge = idx
                else:
                    if speedup > max_force_unroll:
                        max_force_unroll = speedup
                        argmax_force_unroll = idx
            if max_unmerge < 1.005 or max_unmerge <= max_force_unroll + 0.005:
                rows_to_delete.append(index)
            else:
                diffs.append(max_unmerge - max_force_unroll)
                arg_maxs.append((argmax_unmerge, argmax_force_unroll))
                max_unmerge = round(max_unmerge, args.precision)
                max_force_unroll = round(max_force_unroll, args.precision)
                speedups.append((max_unmerge, max_force_unroll))
        print("deleting rows: ", rows_to_delete)
        result_df.drop(result_df.index[rows_to_delete], inplace=True)
        if len(diffs) == 0:
            print("No interesting loops found")
            quit()
        else:
            result_df['diff'] = diffs
            speedup_unmerge, speedups_unroll = list(zip(*speedups))
            argmax_unmerge, argmax_unroll = list(zip(*arg_maxs))
            result_df['best_unmerge'] = list(zip(speedup_unmerge, argmax_unmerge))
            result_df['best_unroll'] = list(zip(speedups_unroll, argmax_unroll))
            configs = []
            if args.late:
                configs = ['tuunmerge-late', 'unmerge-late', 'unroll-late']
            elif args.no_late:
                configs = ['O3-disable', 'tuunmerge', 'tuunmerge-noSubUnroll', 'unroll', 'unmerge',
                           'unmerge-noSubUnroll']
            else:
                print("ERROR: need to specify --late or --no-late")
                quit()
            for index, row in result_df.iterrows():
                csv_row = []
                csv_row.append(args.name)
                csv_row.append(unrolling)
                loop_name = row['names']
                csv_row.append(loop_name)
                found_at_least_one = False
                for config in configs:
                    if config in name_2_idx and name_2_idx[config] in result_df.columns:
                        csv_row.append(round(row[name_2_idx[config]], args.precision))
                        found_at_least_one = True
                    else:
                        csv_row.append(-1)
                if found_at_least_one:
                    csv_row.append(idx_2_name[row['best_unmerge'][1]])
                    csv_row.append(idx_2_name[row['best_unroll'][1]])
                    csv_row.append(round(row['diff'], args.precision))
                    with open(args.interesting, 'a') as f:
                        f.write(";".join([str(x) for x in csv_row]) + "\n")

    return result_df


def get_num_unrolling(filename):
    return filename[filename.index("_") + 1: get_index(filename, '_', 2)]


def get_confidence_interval(exec_times):
    # scipy requires blas and lapack which makes it hard to install on some systems
    # As a workaround, we just return 0, 0 if scipy is not installed since we don't use the confidence intervals in our paper
    if len(exec_times) == 1:
        return (0, 0)
    try:
        import scipy.stats as st
        conf_interval = st.t.interval(0.95, len(exec_times) - 1, loc=np.mean(exec_times), scale=st.sem(exec_times))
        return (round(conf_interval[0], 4), round(conf_interval[1], 4))
    except:
        return (0, 0)


def drop_uninteresting_data(df_metadata):
    df_metadata.drop('startCol', axis=1, inplace=True)
    df_metadata.drop('endCol', axis=1, inplace=True)
    df_metadata.drop('startIsImplicitCode', axis=1, inplace=True)
    df_metadata.drop('endIsImplicitCode', axis=1, inplace=True)


def add_metadata_to_df(df, metadata):
    # join execution_time df and meta_data df if meta data exists
    if metadata is not None:
        df_metadata = pd.read_csv(metadata)
        drop_uninteresting_data(df_metadata)
        df = df.merge(df_metadata, how='left', left_on=['loopIdx'], right_on=['loopIdx'])
    return df


def get_unrolled_loops(name):
    return get_loops_and_unroll_factors(name)[0]


def get_loops_and_unroll_factors(name):
    prefix = 'heuristic::'
    if name == prefix:
        # heuristic decided to not u&u any loop
        return [], []
    name = name[len(prefix):]
    name_split = name.split(',')
    loop_indices = [int(x.split('_')[0]) for x in name_split]
    unroll_factors = [int(x.split('_')[1]) for x in name_split]
    return loop_indices, unroll_factors


def df_create_stats(df, use_nvprof=False, nvprof_metric="ALL_KERNELS", app_name=None):
    default_all_kernel_times = []
    loop_key_2_loop_data = get_loop_data(app_name, default_all_kernel_times, df, nvprof_metric, use_nvprof)

    loops = loop_key_2_loop_data.keys()
    default_loop_data = None
    idx = 0
    for loop in loops:
        idx += 1
        loop_data = loop_key_2_loop_data[loop]
        compute_stats(loop, loop_data)
        if loop[1] == 'default':
            if default_loop_data is not None:
                print("ERROR: multiple default loops found")
                quit()
            default_loop_data = loop_data

    default_code_size = default_loop_data.code_size
    default_exec_time = np.median(default_loop_data.execution_times)
    speedups = get_speedups(app_name, default_all_kernel_times, default_exec_time, loops, nvprof_metric, use_nvprof)
    codesize_factors = get_codesize_factors(loops, default_code_size, loop_key_2_loop_data)

    columns = ['loopIdx', 'name', 'numUnrolling', 'avg', 'median', 'std', 'conf_interval', 'comp_time', 'speedup',
               'codesize', 'codesize_factor', 'filename', 'numIterations']
    column_2_data = {'loopIdx': loop_indices, 'name': names, 'numUnrolling': num_unrollings, 'avg': avgs,
                     'median': medians, 'std': stds, 'conf_interval': conf_intervals,
                     'comp_time': median_compilation_times, 'speedup': speedups, 'codesize': codesizes,
                     'codesize_factor': codesize_factors, 'filename': filenames, 'numIterations': num_iterations}

    df_exec_times = pd.DataFrame(column_2_data, columns=columns)

    reset_globals()
    return df_exec_times


def get_speedups(app_name, default_all_kernel_times, default_exec_time, loops, nvprof_metric, use_nvprof):
    speedups = []
    idx = 0
    for loop in loops:
        median = medians[idx]
        idx += 1
        if use_nvprof and nvprof_metric == "LOOP_KERNELS" and not measure_all_kernels(app_name):
            global app_2_loop_2_kernels
            init_kernel_map(app_2_loop_2_kernels)
            loop_idx = loop[3]
            loop_name = loop[1]
            if 'heuristic' in loop_name:
                kernels = set()
                for unrolled_loop_idx in get_unrolled_loops(loop_name):
                    for kernel in get_kernels(unrolled_loop_idx, app_name):
                        kernels.add(kernel)
            else:
                kernels = get_kernels(loop_idx, app_name)

            default_times = []
            for default_time in default_all_kernel_times:
                sum_kernel_times = 0
                for kernel in kernels:
                    sum_kernel_times += default_time[kernel]
                default_times.append(sum_kernel_times)
            default_time = np.median(default_times)
            speedup = default_time / median
            speedups.append(speedup)
        else:
            speedup = default_exec_time / median
            speedups.append(speedup)
    return speedups


def get_loop_data(app_name, default_all_kernel_times, df, nvprof_metric, use_nvprof):
    loop_key_2_loop_data = {}
    for _, row in df.iterrows():
        filename = row['filename']
        name = row['name']
        num_unrolling = row['numUnrolling']
        loop_idx = row['loopIdx']
        if use_nvprof:
            nvprof_log = row['perf_log']
            loop_idx_list = [loop_idx]
            if 'heuristic' in name:
                loop_idx_list = get_unrolled_loops(name)

            execution_time = nvprof_get_time(nvprof_log, nvprof_metric, app_name, loop_idx_list)
            if loop_idx == DEFAULT_LOOP_IDX and nvprof_metric == "LOOP_KERNELS" and not measure_all_kernels(app_name):
                default_all_kernel_times.append(execution_time)
                execution_time = 1
        else:
            execution_time = row['execution_time']
        compilation_time = row['compilation_time']
        codesize = row['codesize']

        loop = (filename, name, num_unrolling, loop_idx)
        loop_object = LoopData()
        if loop in loop_key_2_loop_data:
            loop_object = loop_key_2_loop_data[loop]
        else:
            loop_key_2_loop_data[loop] = loop_object
        loop_object.add_execution_time(execution_time)
        loop_object.add_compilation_time(compilation_time)
        loop_object.loop_idx = loop_idx
        loop_object.code_size = codesize

    return loop_key_2_loop_data


def get_codesize_factors(loops, default_code_size, loop_key_2_loop_data):
    # need to maintain order which is why we first compute codesize factor for all non default programs
    codesize_factors = []
    for loop in loops:
        if loop[1] == 'default':
            codesize_factors.append(1.0)
            continue
        loop_data = loop_key_2_loop_data[loop]
        codesize = loop_data.code_size
        codesize_factor = codesize / float(default_code_size)
        codesize_factors.append(codesize_factor)
    return codesize_factors


def compute_stats(loop, loop_data):
    num_iterations.append(len(loop_data.execution_times))
    median = np.median(loop_data.execution_times)
    medians.append(median)
    average = np.average(loop_data.execution_times)
    avgs.append(average)
    stds.append(np.std(loop_data.execution_times) / average)
    conf_intervals.append(get_confidence_interval(loop_data.execution_times))
    loop_indices.append(loop_data.loop_idx)
    median_compilation_times.append(np.median(loop_data.compilation_times))
    codesizes.append(loop_data.code_size)
    filenames.append(loop[0])
    names.append(loop[1])
    num_unrollings.append(loop[2])


def sort_df(args, df):
    print(args)
    df = df.sort_values([args.sort_by], ascending=[False])
    return df


def drop_if_exists(df, column):
    if column in df.columns:
        df.drop(column, axis=1, inplace=True)


def compact_representation(args, df):
    if not args.full_output:
        # check if column exists before dropping
        drop_if_exists(df, 'sizeIsValid')
        drop_if_exists(df, 'containsUseOutsideLoop')
        drop_if_exists(df, 'containsPHI')
        drop_if_exists(df, 'exitBlocksContainPHI')
        drop_if_exists(df, 'containsBarrier')
        drop_if_exists(df, 'duplicatable')


def get_index(word, char, n):
    # returns the index of the n-th occurrence of char in word
    num_seen = 0
    for i, c in enumerate(word):
        if c == char:
            num_seen += 1
            if num_seen == n:
                return i
    return -1


def filename_2_config(filename):
    config = filename.split('_')[-2]
    return config


def datadump_pretty(datadump):
    datadump = datadump.split("/")[-1]
    split_undescore = datadump.split("_")
    config = split_undescore[-2]
    unrolling = str(int(split_undescore[-1][:-4]))

    if config == 'uu':
        return 'u&u ' + unrolling
    if config == 'unmerge':
        return 'unmerge'
    return config + " " + unrolling


def get_app_name(app):
    if '/' in app:
        app = app.split('/')[-1]
    return app

def plot_O3_scatter(datadumps, other_datadumps, column_name='speedup', compare_to='unroll', disable_plot=False, results_dir='./'):
    columns = ['app', 'loop', 'speedup', 'loopSize', 'codeSizeIncrease', 'comp_time', 'codesize', 'std']

    datadump_2_df = {}
    all_datadumps = datadumps + other_datadumps
    for datadump in all_datadumps:
        print("Reading: ", datadump)
        df = pd.read_csv(datadump, sep=";", names=columns)
        datadump_2_df[datadump] = df

    rows = []
    datadump_2_loop_2_speedup = {}

    for datadump, df in datadump_2_df.items():
        datadump_2_loop_2_speedup[datadump] = {}
        for index, row in df.iterrows():
            application_name = get_app_name(row['app'])
            loop = row['loop']
            value = row[column_name]
            loop_id = str(application_name) + '_' + str(loop)
            datadump_2_loop_2_speedup[datadump][loop_id] = value

    all_loops = []
    for i in range(len(datadumps)):
        datadump = datadumps[i]
        other_datadump = other_datadumps[i]
        loop_2_speedup = datadump_2_loop_2_speedup[datadump]
        other_loop_2_speedup = datadump_2_loop_2_speedup[other_datadump]
        for loop_id, value in loop_2_speedup.items():
            if loop_id in other_loop_2_speedup:
                rows.append((loop_id, value, other_loop_2_speedup[loop_id],
                             datadump_pretty(datadump) + ' vs ' + datadump_pretty(other_datadump)))
                all_loops.append((loop_id, value, other_loop_2_speedup[loop_id]))

    df = pd.DataFrame(rows, columns=['loop', 'u&u', compare_to, 'configs'])

    sns.set(style="whitegrid")
    sns.set_palette('colorblind')
    fig = plt.figure(figsize=(18, 12))

    # get min and max speedup to set axis limits
    min_speedup = min([x[1] for x in all_loops])
    max_speedup = max([x[1] for x in all_loops])
    min_lim = min_speedup * 0.9
    max_lim = max_speedup * 1.1
    plt.xlim(min_lim, max_lim)
    plt.ylim(min_lim, max_lim)

    marker_size = 450
    ax = sns.scatterplot(data=df, x='u&u', y=compare_to, hue='configs', s=marker_size)
    plt.plot([min_lim, max_lim], [min_lim, max_lim], color='r')
    ax.set_xlabel('speedup: u&u', fontsize=32)
    ax.set_ylabel('speedup: ' + compare_to, fontsize=32)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)

    # create legend
    lgnd = ax.legend(loc='upper center', fontsize=32, bbox_to_anchor=(0.5, 1.12), ncol=3, fancybox=True, shadow=True,
                     labelspacing=0.2, columnspacing=0.05, handletextpad=0.01)
    try:
        for handle in lgnd.legendHandles:
            # handle.set_sizes([marker_size])
            handle._legmarker.set_markersize(marker_size)
    except:
        print("WARNING: Could not set marker size")

    # set file name
    name = 'fig8'
    if compare_to == 'unroll':
        name += 'a'
    elif compare_to == 'unmerge':
        name += 'b'
    else:
        print("Unknown compare_to: ", compare_to)
        quit()

    plt.savefig(results_dir + name + '.pdf', format='pdf', bbox_inches='tight')

    if not disable_plot:
        plt.show()


def cmp_speedup_size(execution_time_log, out_file):
    df = pd.read_csv(execution_time_log, sep=';')
    # df = df_create_stats(df, use_nvprof=False)
    df = df_create_stats(df, use_nvprof=True, nvprof_metric="ALL_KERNELS", app_name=args.name)
    first_file = execution_time_log.split('/')[-1]
    first_underscore = first_file.index('_')
    source_filename = first_file[:first_underscore]
    target_triple = first_file[get_index(first_file, '_', 2) + 1: get_index(first_file, '_', 3)]
    metadata_early = source_filename + "_" + target_triple + "_metadata.csv"
    metadata_late = source_filename + "_" + target_triple + "_late_metadata.csv"
    metadata = metadata_late if "late" in execution_time_log else metadata_early
    df = add_metadata_to_df(df, metadata)
    lines = []
    for idx, row in df.iterrows():
        name = row['name']
        speedup = row['speedup']
        loopSize = row['loopSize']
        codeSizeIncrease = row['codesize_factor']
        compileTime = row['comp_time']
        codeSize = row['codesize']
        std = row['std']
        lines.append(
            args.name + ";" + name + ";" + str(speedup) + ";" + str(loopSize) + ";" + str(codeSizeIncrease) + ";" + str(
                compileTime) + ";" + str(codeSize) + ";" + str(std))
    if len(lines) != 0:
        with open(out_file, "a") as f:
            f.write("\n".join(lines) + "\n")


def get_apps():
    apps = ['bezier-surface', 'bn', 'bspline-vgh', 'ccs', 'clink', 'complex', 'contract', 'coordinates', 'haccmk',
            'lavamd', 'rainflow', 'libor', 'mandelbrot', 'qtclustering', 'quicksort', 'xsbench']
    return apps


def nvprof_get_time(nvprof_output, nvprof_metric=None, app_name=None, loop_indices=[]):
    df = parse_nvprof(nvprof_output, nvprof_metric, app_name, loop_indices)
    if nvprof_metric == "LOOP_KERNELS" and not measure_all_kernels(app_name):
        if len(loop_indices) == 1:
            loop_idx = loop_indices[0]
            if loop_idx == DEFAULT_LOOP_IDX:
                # return dict with kernel names as keys and time as values
                kernel_2_time = {}
                for idx, row in df.iterrows():
                    kernel_2_time[row['Name']] = row['Time']
                return kernel_2_time

    return nvprof_sum_time(df)


def parse_nvprof(nvprof_output, nvprof_metric="ALL_KERNELS", app_name=None, loop_indices=[]):
    lines = []
    start_parsing = False

    # read nvprof output line by line
    try:
        with open(nvprof_output) as file:
            read_nvprof(lines, start_parsing, file)
    except FileNotFoundError:
        # In case table is not created on same system as the measurements were performed
        # try to find the file in the current working directory
        print("File not found:", nvprof_output)
        nvprof_filename = nvprof_output.split('/')[-1]
        alternative_path = os.getcwd() + "/nvprof/" + nvprof_filename
        print("Trying alternative path:", alternative_path)
        with open(alternative_path) as file:
            print("Found file at alternative path")
            read_nvprof(lines, start_parsing, file)

    # remove second line (it looks like this: ,%,s,,ms,ms,ms,)
    line = lines.pop(1)
    line_split = line.split(',')
    time_unit = line_split[2]
    multiplier = 1
    if time_unit == 'ms':
        multiplier = 1
    elif time_unit == 's':
        multiplier = 1000
    elif time_unit == 'us':
        multiplier = 0.001
    elif time_unit == 'ns':
        multiplier = 0.000001
    else:
        print("ERROR: Unknown time unit: ", time_unit)
        quit()

    # create pandas dataframe from lines
    df = pd.read_csv(StringIO('\n'.join(lines)), sep=',')
    df = df.astype(
        {"Time(%)": 'float64', "Time": 'float64', "Calls": int, "Avg": 'float64', "Min": 'float64', "Max": 'float64'})
    # multiply time column with multiplier
    df['Time'] = df['Time'] * multiplier
    df = nvprof_filter_kernels(df, nvprof_metric, app_name, loop_indices)
    return df

def read_nvprof(lines, start_parsing, file):
    for line in file:
        line = line.rstrip()
        if not start_parsing:
            if line == '"Type","Time(%)","Time","Calls","Avg","Min","Max","Name"':
                start_parsing = True
                lines.append(line)
        else:
            lines.append(line)
    return start_parsing


def nvprof_filter_kernels(df, nvprof_metric, app_name=None, loop_indices=[]):
    # remove all rows with Type != GPU activities
    df = df[df['Type'] == 'GPU activities']

    if nvprof_metric != "END2END":
        # remove memory transfer times
        df = df[df['Name'] != '[CUDA memcpy HtoD]']
        df = df[df['Name'] != '[CUDA memcpy DtoH]']
        df = df[df['Name'] != '[CUDA memset]']

    if nvprof_metric == "LOOP_KERNELS":
        if not measure_all_kernels(app_name):
            if len(loop_indices) >= 1 and loop_indices[0] != DEFAULT_LOOP_IDX:
                kernels = set()
                for loop_idx in loop_indices:
                    print("loop_idx: ", loop_idx)
                    for kernel in get_kernels(loop_idx, app_name):
                        kernels.add(kernel)
                df = df[df['Name'].isin(kernels)]

    return df


def nvprof_sum_time(df):
    # print(df.to_string())
    sum_time = df['Time'].sum()
    return sum_time


def add_line_breaks(s):
    app_2_pretty = {}
    app_2_pretty['atomicAggregate'] = 'atomic\naggregate'
    app_2_pretty['bezier-surface'] = 'bezier\nsurface'
    app_2_pretty['bspline-vgh'] = '      bspline\n      vgh'
    app_2_pretty['coordinates'] = 'coor-\ndinates'
    app_2_pretty['contract'] = 'contract'
    app_2_pretty['mandelbrot'] = 'mandel-\nbrot'
    app_2_pretty['rainflow'] = 'rainflow'
    app_2_pretty['xsbench'] = 'XSBench'
    app_2_pretty['lavamd'] = 'lavaMD'
    app_2_pretty['quicksort'] = 'quicksort'
    app_2_pretty['qtclustering'] = 'qtclus-\ntering'
    if s in app_2_pretty:
        return app_2_pretty[s]

    result = ''
    for i, c in enumerate(s):
        result += c
        if (i + 1) % 7 == 0 and (i + 1) != len(s):
            result += '-\n'
    return result


def get_table_line(execution_time, app_name, get_baseline):
    df = pd.read_csv(execution_time, sep=";")
    kernel_name_2_times = defaultdict(list)
    for idx, row in df.iterrows():
        name = row['name']
        if get_baseline and name != 'default':
            continue
        elif not get_baseline and 'heuristic' not in name:
            continue
        nvprof_log = row['perf_log']
        nvprof_df = parse_nvprof(nvprof_log, nvprof_metric="END2END")

        # nsys truncates kernel names so different kernels can have the same name
        # we first collect all times for one run and then sum the times up
        one_run_kernel_name_2_times = defaultdict(list)
        for idx, row in nvprof_df.iterrows():
            name = row['Name']
            time = row['Time']
            one_run_kernel_name_2_times[name].append(time)

        # sum up times for each kernel
        for name, times in one_run_kernel_name_2_times.items():
            kernel_name_2_times[name].append(sum(times))

    sum_means = 0
    sum_compute_list = []
    sum_compute = 0
    for kernel, times in kernel_name_2_times.items():
        sum_means += np.mean(times)
        if not kernel.startswith('[CUDA'):
            sum_compute += np.mean(times)
            sum_compute_list.append(times)

    compute_sums_for_each_run = []
    for run in range(len(sum_compute_list[0])):
        sum_run = 0
        for kernel in range(len(sum_compute_list)):
            sum_run += sum_compute_list[kernel][run]
        compute_sums_for_each_run.append(sum_run)

    compute_kernel_mean = np.mean(compute_sums_for_each_run)
    compute_kernel_std = np.std(compute_sums_for_each_run)
    compute_kernel_relative_std = compute_kernel_std / compute_kernel_mean * 100

    compute_percent = sum_compute / sum_means * 100
    return (app_name, compute_percent, compute_kernel_mean, compute_kernel_relative_std)

def table_kernel_times(execution_time, app_name, output_csv):
    (baseline_app_name, baseline_comp_percent, baseline_compute_kernel_mean, baseline_compute_kernel_relative_std) = get_table_line(execution_time, app_name, get_baseline=True)
    (heuristic_app_name, heuristic_comp_percent, heuristic_compute_kernel_mean, heuristic_compute_kernel_relative_std) = get_table_line(execution_time, app_name, get_baseline=False)
    if baseline_app_name != heuristic_app_name:
        print("ERROR: baseline_app_name != heuristic_app_name")
        quit()

    num_decimal_places = 2
    line = "{app_name};{baseline_compute_percent}%;{baseline_compute_mean} ms;+/-{baseline_compute_rsd}%;{heuristic_compute_mean} ms;+/-{heuristic_compute_rsd}%".format(app_name=baseline_app_name,
                                                                            baseline_compute_percent=round(baseline_comp_percent, num_decimal_places),
                                                                            baseline_compute_mean=round(baseline_compute_kernel_mean, num_decimal_places),
                                                                            baseline_compute_rsd=round(baseline_compute_kernel_relative_std, num_decimal_places),
                                                                            heuristic_compute_mean=round(heuristic_compute_kernel_mean, num_decimal_places),
                                                                            heuristic_compute_rsd=round(heuristic_compute_kernel_relative_std, num_decimal_places))
    lines = []
    lines.append(line)
    with open(output_csv, 'a') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    args = parseArguments()
    pd.set_option('display.precision', args.precision)

    if args.table is not None:
        table_kernel_times(args.execution_times[0], args.name, args.table)
        quit()

    if args.cmp_size is not None:
        if len(args.execution_times) != 1:
            print("ERROR: Please provide a csv file with exactly one execution time log for the comparison with O3")
            quit()
        cmp_speedup_size(args.execution_times[0], args.cmp_size)
        quit()

    if args.plot_O3_scatter is not None:
        listOdd = args.execution_times[1::2]
        listEven = args.execution_times[::2]
        plot_O3_scatter(listEven, listOdd, column_name='speedup', compare_to=args.plot_O3_scatter,
                        disable_plot=args.disable_plot, results_dir=args.plot_O3_scatter_dir)
        quit()

    if args.compareSpeedup:
        df = create_df_compare(args)
    else:
        df = create_df(args)
        enable_nvprof = args.use_nvprof is not None
        df = df_create_stats(df, enable_nvprof, nvprof_metric=args.use_nvprof, app_name=args.name)
        df = sort_df(args, df)

    df = add_metadata_to_df(df, args.metadata)
    if args.sort_reverse:
        df = df[::-1]

    compact_representation(args, df)

    if not args.heuristicFullName and 'name' in df.columns:
        df['name'] = df['name'].apply(lambda x: 'heuristic' if x.startswith('heuristic') else x)

    print(df.to_string(index=False))

    if args.o is not None:
        df.to_csv(args.o, index=False)
