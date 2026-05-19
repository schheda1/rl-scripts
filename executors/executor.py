import argparse
import os
import subprocess
import time
import uuid
from collections import defaultdict
from io import StringIO
from pathlib import Path
from util import *

import pandas as pd


class HeuristicInfo():
    def __init__(self, unrolled_loops=[]):
        self.unrolled_loops = unrolled_loops


    def print(self):
        print("- The heuristic decided to unroll the following loops:")
        for (loopIdx, unroll_factor) in self.unrolled_loops:
            paper_unroll_factor = int(unroll_factor)
            print("-- Loop idx: ", loopIdx, " Unroll factor: ", paper_unroll_factor)


    def to_string(self):
        return ",".join(["{loopIdx}_{unroll_factor}".format(loopIdx=x, unroll_factor=y) for (
            x, y) in self.unrolled_loops])

class Executor():
    def __init__(self, name, unrollings=[1], num_iterations=1, logs_dir='', include_files=None, exclude_files=None, timeout_limit='300', skip_loops=[], start_loop=None, end_loop=None, compile_configs=['uu'], build_dir='./'):
        self.name = name
        self.unrollings = unrollings
        self.num_iterations = num_iterations
        self.logs_dir = logs_dir
        self.include_files = include_files
        self.exclude_files = exclude_files
        self.timeout_limit = timeout_limit
        self.skip_loops = skip_loops
        self.start_loop = start_loop
        self.end_loop = end_loop
        self.compile_configs = compile_configs
        self.filter_branchless = False
        self.build_dir = build_dir


    def log_failure(self, num_unrolling, filename, name, loopIdx, compilation_time, execution_time, target_triple, config):
        # If a compilation or execution fails, we log the failure
        path_to_log_file_template = "{logs_dir}{name}/{filename}_{num_unrolling}_{target_triple}_{config}_failures.txt"
        path_to_log_file = path_to_log_file_template.format(logs_dir=self.logs_dir, name=self.name, filename=filename, num_unrolling=num_unrolling, target_triple=target_triple, config=config)
        log_file = Path(path_to_log_file)
        log_file_exists = log_file.is_file()
        with open(log_file, 'a+') as f:
            # write header if file does not exist
            if not log_file_exists:
                f.write(
                    'loopIdx;name;numUnrolling;execution_time;compilation_time;filename;target_triple\n')
            entries = [str(loopIdx), name, str(num_unrolling), str(
                execution_time), str(compilation_time), filename, target_triple]
            output_line = ';'.join(entries) + '\n'
            f.write(output_line)


    def log_uncompilable(self, cfile, df_loop_data, target_triple):
        # We keep track of loops that we do not want to compile (e.g. loops without branches)
        outdir = self.logs_dir + self.name + '/' + 'uncompilable/'
        loop_metadata_file_path = outdir + cfile + \
            '_' + target_triple + '_metadata.csv'
        loop_metadata_file = Path(loop_metadata_file_path)
        loop_metadata_file_exists = loop_metadata_file.is_file()
        if not loop_metadata_file_exists:
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            df_loop_data.to_csv(loop_metadata_file_path, index=False)

    def parse_simple_heuristic_output(self, output):
        # output looks like this:
        # UnrollAndUnmergeHeuristic::0;2
        # UnrollAndUnmergeHeuristic::1;2
        prefix = 'UnrollAndUnmergeHeuristic::'
        prefix_len = len(prefix)
        unrolled_loops = []
        for line in output.splitlines():
            if line.startswith('UnrollAndUnmergeHeuristic'):
                line_substring = line[prefix_len:]
                line_split = line_substring.split(";")
                loopIdx = line_split[0]
                unroll_factor = line_split[1]
                unrolled_loops.append((loopIdx, unroll_factor))
        return HeuristicInfo(unrolled_loops)


    def build(self, additional_cflags, failed_indices, num_unrolling, filename, name, loop_idx, row_idx, target_triple, config):
        # This function compiles the program with the given configuration
        compilation_was_sucessful = True

        print("Compiling", prettify_name(name, config, num_unrolling))
        t1 = time.perf_counter()
        (rc, out, err) = self.make(
            additional_cflags=additional_cflags, withTimeout=True)
        t2 = time.perf_counter()
        compilation_time = t2 - t1
        print("- Compilation time:", round(compilation_time, 2), "s")

        # get data output by heuristic during compilation
        heuristic_info = self.parse_simple_heuristic_output(err)
        if 'uu-heuristic' in name:
            heuristic_info.print()

        if rc != 0:
            # compilation failed
            failed_indices.append(row_idx)
            self.log_failure(num_unrolling, filename,
                             name, loop_idx, compilation_time, -1, target_triple, config)
            compilation_was_sucessful = False
            codeSize = -1
        else:
            # measure code size and return code size
            codeSize = self.get_code_size()
            self.exec(['cp', self.get_executable_name(),
                      self.get_executable_name()+name])
            
        print('- Codesize: {} bytes'.format(codeSize))
        return (compilation_was_sucessful, compilation_time, codeSize, heuristic_info)


    def get_code_size(self):
        (rc, out, err) = self.exec(['size', self.get_executable_name()])
        if rc != 0:
            return -2
        splitlines = out.splitlines()
        line = splitlines[1]
        text = line.split('\t')[0]
        codesize = int(''.join(text.split()))
        return codesize


    def perf_command(self, num_unrolling, filename, name, target_triple, config):
        id = uuid.uuid4().hex
        log_file = self.nvprof_csv_filename(
            num_unrolling, filename, name, target_triple, config, id)
        return ('nvprof --csv --log-file {log_file} '.format(log_file=log_file), log_file)


    def nsys_command(self):
        return 'nsys profile --stats=true '


    def get_simple_heuristic_name(self, heuristic_info):
        name = 'heuristic::' + ",".join(["{loopIdx}_{unroll_factor}".format(loopIdx=x, unroll_factor=y) for (
            x, y) in heuristic_info.unrolled_loops])
        return name


    def profile_with_nsys(self, perf_log_file):
        nsys_command = self.nsys_command()
        
        # restart timer
        t1 = time.perf_counter()
        (rc, out, err) = self.exec(
            nsys_command + self.execute_command(), shell=True)
        
        output_to_parse = None
        # check if out contains 'gpumemtime'
        if contains_parseable_data(out):
            output_to_parse = out
        elif contains_parseable_data(err):
            output_to_parse = err
            rc = 0
        else:
            print("nsys output not detected in out or err")
            print("out:", out)
            print("err:", err)
            rc = -1

        nsys_output = parse_nsys_output(output_to_parse.splitlines())
        # write nsight output to log file
        with open(perf_log_file, 'w+') as f:
            f.write(nsys_output)
        return (rc, out, err, t1)


    def measure(self, failed_indices, idx, num_unrolling, filename, name, loop_idx, compilation_time, exec_time_dict, codesize, target_triple, config, heuristic_info):
        # Execute the program with the given configuration and measure the execution time
        print("- Executing", prettify_name(name, config, num_unrolling))

        curr_binary_name = self.get_executable_name() + name

        self.exec(['cp', curr_binary_name, self.get_executable_name()])
        t1 = time.perf_counter()

        (perf_command, perf_log_file) = self.perf_command(
            num_unrolling, filename, name, target_triple, config)
        (rc, out, err) = self.exec(
            perf_command + self.execute_command(), shell=True)
        
        if rc != 0:
            # if nvprof fails, we try nsys because nvprof is not supported on compute capability >= 8.0
            (rc, out, err, t1) = self.profile_with_nsys(perf_log_file)

        t2 = time.perf_counter()
        runtime = self.get_exec_time_from_output(out)
        if runtime < 0:
            runtime = t2 - t1

        isCorrect = self.validate_output(out)
        if not isCorrect or rc != 0:
            # run failed
            print("!!!Run failed!!!")
            print("err:", err)
            print("out:", out)
            print("return code: ", rc)
            print("Validation successful: ", isCorrect)
            failed_indices.append(idx)

            if 'uu-heuristic' in name:
                name = self.get_simple_heuristic_name(heuristic_info)
            self.log_failure(num_unrolling, filename,
                             name, loop_idx, compilation_time, runtime, target_triple, config)
            return False

        # write to file
        exec_time_dict[(name, loop_idx)].append(runtime)
        if 'uu-heuristic' in name:
            name = self.get_simple_heuristic_name(heuristic_info)
        self.log_execution_time(
            num_unrolling, filename, compilation_time, name, loop_idx, runtime, codesize, err, target_triple, config, perf_log_file, heuristic_info)
        return True

    def execute_and_validate_checksum(self, df_loop_data, num_unrolling, cfilename, filename, target_triple, config):
        # Compile, execute and measure

        # Maps (name, loopIdx) to list of execution times
        exec_time_dict = defaultdict(list)

        # Keep track of loops that we could not compile
        failed_indices = []

        # Maps name to compilation time
        name2compilation_time = defaultdict(float)

        # Maps name to codesize
        name2codesize = defaultdict(int)
        
        # Maps name to heuristic info
        name2heuristic_info = {}

        # Compile baseline, heuristic (if requested) and all loops
        self.compile(df_loop_data, num_unrolling, cfilename,
                     filename, failed_indices, name2compilation_time, name2codesize, name2heuristic_info, target_triple, config)

        # Measure self.num_iterations times
        for iteration in range(self.num_iterations):
            print("Iteration:", iteration)
            name = 'default'
            loop_idx = DEFAULT_LOOP_IDX
            # execute default
            if not loop_idx in failed_indices:
                self.measure(failed_indices, -1, num_unrolling, filename, name, loop_idx,
                             name2compilation_time[name], exec_time_dict, name2codesize[name], target_triple, config, name2heuristic_info[name])
                
            # If we don't want to perform measurements for the heuristic, we perform measurements for each loop
            if not "uu-heuristic" in config:
                for idx, row in df_loop_data.iterrows():
                    if idx in failed_indices:
                        continue
                    loop_idx = row['loopIdx']
                    name = 'loop-' + str(loop_idx)
                    self.measure(failed_indices, idx, num_unrolling, filename, name, loop_idx,
                                 name2compilation_time[name], exec_time_dict, name2codesize[name], target_triple, config, name2heuristic_info[name])
            else:
                name = config
                loop_idx = HEURISTIC_LOOP_IDX
                if not loop_idx in failed_indices:
                    self.measure(failed_indices, -HEURISTIC_LOOP_IDX, num_unrolling, filename, name, loop_idx,
                                 name2compilation_time[name], exec_time_dict, name2codesize[name], target_triple, config, name2heuristic_info[name])

        return exec_time_dict

    def compile(self, df_loop_data, num_unrolling, cfilename, filename, failed_indices, name2compilation_time, name2codesize, name2heuristic_info, target_triple, config):
        if not "uu-heuristic" in config:
            # Compile the application once for each loop
            for idx, row in df_loop_data.iterrows():
                loop_idx = row['loopIdx']
                additional_cflags = self.getPassFlags(
                    config, num_unrolling, cfilename, [loop_idx], target_triple)
                name = 'loop-' + str(loop_idx)
                (successful, compilation_time, codesize, heuristic_info) = self.build(additional_cflags,
                                                                                      failed_indices, num_unrolling, filename, name, loop_idx, idx, target_triple, config)
                if successful:
                    name2compilation_time[name] = compilation_time
                    name2codesize[name] = codesize
                    name2heuristic_info[name] = heuristic_info

        # build default
        name = 'default'
        loop_idx = DEFAULT_LOOP_IDX
        (successful, compilation_time, codesize, heuristic_info) = self.build(
            '', failed_indices, num_unrolling, filename, name, loop_idx, loop_idx, target_triple, config)
        if successful:
            name2compilation_time[name] = compilation_time
            name2codesize[name] = codesize
            name2heuristic_info[name] = heuristic_info

        if "uu-heuristic" in config:
            name = config
            loop_idx = HEURISTIC_LOOP_IDX
            additional_cflags = self.getPassFlagsHeuristic(
                config, num_unrolling, cfilename, target_triple)
            (successful, compilation_time, codesize, heuristic_info) = self.build(additional_cflags,
                                                                                  failed_indices, num_unrolling, filename, name, loop_idx, loop_idx, target_triple, config)
            if successful:
                name2compilation_time[name] = compilation_time
                name2codesize[name] = codesize
                name2heuristic_info[name] = heuristic_info

    def getPassFlagsHeuristic(self, config, num_unrolling, cfilename, target_triple):
        flags = self.getPassFlags(config, num_unrolling, cfilename, [], target_triple)
        return flags

    def getPassFlags(self, config, num_unrolling, cfilename, loop_indices, target_triple):
        if config == "unroll":
            return self.getPassFlagsUnroll(num_unrolling, cfilename, loop_indices, target_triple)
        elif config == "uu":
            return self.getPassFlagsUU(num_unrolling, cfilename, loop_indices, target_triple)
        elif config == "unmerge":
            return self.getPassFlagsUnmerge(cfilename, loop_indices, target_triple)
        elif config == "uu-heuristic":
            return "-mllvm --enable-uu-heuristic"
        else:
            print("Unknown config: ", config)
            exit(1)

    def getPassFlagsUnroll(self, num_unrolling, cfilename, loop_indices, target_triple):
        loop_indices = [str(x) for x in loop_indices]
        enable_pass = '-mllvm --enable-unroll'
        unrolling_arg = '-mllvm --force-unroll-unrollfactor={num_unrolling}'.format(
            num_unrolling=num_unrolling)
        opt_loop_idx_arg = '-mllvm -unroll-opt-loop-idx={loop_indices}'.format(
            loop_indices=",".join(loop_indices))
        filename_arg = '-mllvm --unroll-match-filename={filename}'.format(
            filename=cfilename)
        target_triple_arg = '-mllvm --unroll-match-targettriple={target_triple}'.format(
            target_triple=target_triple)
        additional_cflags = '{enable_pass} {unrolling} {filename_match} {opt_loop_idx} {target_triple_arg}'.format(enable_pass=enable_pass,
                                                                                                                   unrolling=unrolling_arg, opt_loop_idx=opt_loop_idx_arg, filename_match=filename_arg, target_triple_arg=target_triple_arg)
        return additional_cflags

    def getPassFlagsUU(self, num_unrolling, cfilename, loop_indices, target_triple, enable_heuristic=False):
        loop_indices = [str(x) for x in loop_indices]
        enable_pass = '-mllvm --enable-uu'
        unrolling_arg = '-mllvm --uu-unrollfactor={num_unrolling}'.format(
            num_unrolling=num_unrolling)
        opt_loop_idx_arg = '-mllvm -uu-opt-loop-idx={loop_indices}'.format(
            loop_indices=",".join(loop_indices))
        filename_arg = '-mllvm --uu-match-filename={filename}'.format(
            filename=cfilename)
        target_triple_arg = '-mllvm --uu-match-targettriple={target_triple}'.format(
            target_triple=target_triple)
        
        additional_cflags = '{enable_pass} {unrolling} {filename_match} {opt_loop_idx} {target_triple_arg}'.format(enable_pass=enable_pass,
                                                                                                                   unrolling=unrolling_arg, opt_loop_idx=opt_loop_idx_arg, filename_match=filename_arg, target_triple_arg=target_triple_arg)
        return additional_cflags


    def getPassFlagsUnmerge(self, cfilename, loop_indices, target_triple):
        return self.getPassFlagsUU(1, cfilename, loop_indices, target_triple)


    def getPassFlagsDisableUnrolling(self, cfilename, loop_indices, target_triple):
        return self.getPassFlagsTu(0, cfilename, loop_indices, target_triple, False)


    def log_execution_time(self, num_unrolling, filename, compilation_time, name, loopIdx, runtime, codesize, out, target_triple, config, perf_log_file, heuristic_info):
        path_to_log_file = self.logs_dir + self.name + '/' + \
            filename + '_' + str(num_unrolling) + '_' + \
            target_triple + '_' + config + '_times.txt'
        log_file = Path(path_to_log_file)
        log_file_exists = log_file.is_file()

        with open(log_file, 'a+') as f:
            if not log_file_exists:
                f.write(
                    'loopIdx;name;numUnrolling;execution_time;compilation_time;codesize;filename;perf_log' + '\n')

            output_line = str(loopIdx) + ';' + name + ';' + str(num_unrolling) + \
                ';' + str(runtime) + ';' + \
                str(compilation_time) + ';' + \
                str(codesize) + ';' + filename  + \
                ';' + perf_log_file + '\n'
            f.write(output_line)


    def nvprof_csv_filename(self, num_unrolling, filename, name, target_triple, config, id):
        nvprof_dir = self.logs_dir + self.name + '/nvprof/'
        self.mkdir(nvprof_dir)
        path_to_nvprof_csv = nvprof_dir + 'nvprof_' + id + '_' + filename + '_' + \
            str(num_unrolling) + '_' + target_triple + \
            '_' + config + '_' + name + '.csv'
        return path_to_nvprof_csv

    def mkdir(self, dir):
        if not os.path.isdir(dir):
            os.makedirs(dir)


    def execute_command(self):
        pass


    def make_command(self, additional_cflags=''):
        pass


    def get_exec_time_from_output(self, output):
        pass


    def validate_output(self, output):
        pass


    def pre_measurement(self):
        pass


    def post_measurement(self):
        pass


    def get_executable_name(self):
        pass


    def make(self, additional_cflags='', withTimeout=False):
        (rc, out, err) = self.exec(['make', 'clean'])

        if withTimeout:
            timeout_cmd = 'timeout ' + self.timeout_limit + ' '
        else:
            timeout_cmd = ''
        make_command = self.make_command(additional_cflags)
        (rc, out, err) = self.exec("{timeout_cmd}{make_command}".format(
            timeout_cmd=timeout_cmd, make_command=make_command), shell=True)
        return (rc, out, err)


    def exec(self, cmd, shell=False, quit_at_error=False):
        process = subprocess.run(
            cmd, shell=shell, capture_output=True)
        rc = process.returncode
        if rc != 0 and quit_at_error:
            print("Error executing command: ", cmd)
            print(process.stderr.decode('utf-8'))
            print(process.stdout.decode('utf-8'))
            quit()
        return (rc, process.stdout.decode('utf-8'), process.stderr.decode('utf-8'))


    def generate_loop_metadata(self, config):
        print("Looking for loops...")
        # targetTriple -> (name -> metadata)
        target_triple_2_name_2_loop_metadata = defaultdict(str)
        additional_cflags = '-mllvm --enable-loopcount'


        (rc, out, err) = self.make(additional_cflags=additional_cflags)
        filename = ''
        target_triple = ''
        lines = []

        for line in err.splitlines():
            # print(line)
            if line.startswith('LOOPCOUNT METADATA'):
                if filename != '':
                    loop_data = StringIO('\n'.join(lines))
                    try:
                        df = pd.read_csv(loop_data, sep=";")
                        if not target_triple in target_triple_2_name_2_loop_metadata:
                            target_triple_2_name_2_loop_metadata[target_triple] = defaultdict(
                                str)
                        target_triple_2_name_2_loop_metadata[target_triple][filename] = df
                    except Exception as e:
                        print(lines)
                        print(loop_data.getvalue())
                        print(e)
                        quit()
                split_line = line.split(';')
                filename = split_line[1]
                target_triple = split_line[2]
                lines = []
            if line.startswith('LOOPCOUNT::'):
                content = line[11:]
                lines.append(content)

        loop_data = StringIO('\n'.join(lines))
        df = pd.read_csv(loop_data, sep=";")
        if not target_triple in target_triple_2_name_2_loop_metadata:
            target_triple_2_name_2_loop_metadata[target_triple] = defaultdict(
                str)
        target_triple_2_name_2_loop_metadata[target_triple][filename] = df

        # skip certain loops
        if len(self.skip_loops) > 0:
            for target_triple, name_2_loop_metadata in target_triple_2_name_2_loop_metadata.items():
                for filename, df in name_2_loop_metadata.items():
                    for skip_loop_idx in self.skip_loops:
                        df.drop(
                            df[df['loopIdx'] == skip_loop_idx].index, inplace=True)

        # check if there are loops that we might not be able to compile
        for target_triple, name_2_loop_metadata in target_triple_2_name_2_loop_metadata.items():
            for filename, df_loop_data in name_2_loop_metadata.items():
                loops_to_delete = []
                for idx, row in df_loop_data.iterrows():
                    skip_loop = False
                    isDuplicatable = row['duplicatable']
                    if isDuplicatable != 1.0:
                        skip_loop = True
                    containsBarrier = row['containsBarrier']
                    if containsBarrier != 0.0:
                        skip_loop = True
                    functionName = row['function']
                    if functionName == '__ockl_fprintf_append_string_n' or functionName == '__assert_fail' or functionName == '' or functionName is None:
                        skip_loop = True
                    if self.filter_branchless:
                        containsBranch = row['containsBranch']
                        if containsBranch == 0.0:
                            skip_loop = True
                    if skip_loop:
                        loops_to_delete.append(idx)

                df_uncompilable = df_loop_data.loc[loops_to_delete]
                self.log_uncompilable(
                    Path(filename).stem, df_uncompilable, target_triple)
                df_loop_data = df_loop_data.drop(loops_to_delete)
                name_2_loop_metadata[filename] = df_loop_data
        return target_triple_2_name_2_loop_metadata


    def pretty_print_found_loops(self, df):
        df_copy = df.copy()
        df_copy.drop('duplicatable', axis=1, inplace=True)
        df_copy.drop('sizeIsValid', axis=1, inplace=True)
        df_copy.drop('containsPHI', axis=1, inplace=True)
        df_copy.drop('containsUseOutsideLoop', axis=1, inplace=True)
        df_copy.drop('exitBlocksContainPHI', axis=1, inplace=True)
        df_copy.drop('containsBarrier', axis=1, inplace=True)
        print("The following loops have been found:")
        print(df_copy.to_string())


    def execute(self):
        args = self.parseArgs()

        print("Starting measurements for application:", self.name)

        self.pre_measurement()
        self.mkdir(self.logs_dir + self.name)

        for config in self.compile_configs:
            target_triple_2_filename2df_loop_metadata = self.generate_loop_metadata(
                config)
            all_filenames = []
            for target_triple, name_2_loop_metadata in target_triple_2_filename2df_loop_metadata.items():
                for filename in name_2_loop_metadata.keys():
                    if not filename in all_filenames:
                        all_filenames.append(filename)

            if self.include_files is not None:
                all_filenames = [
                    filename for filename in all_filenames if filename in self.include_files]
            if self.exclude_files is not None:
                all_filenames = [
                    filename for filename in all_filenames if filename not in self.exclude_files]
            if args.include_files is not None:
                all_filenames = [
                    filename for filename in all_filenames if filename in args.include_files]
            if args.exclude_files is not None:
                all_filenames = [
                    filename for filename in all_filenames if filename not in args.exclude_files]

            for cfile in all_filenames:
                cfilename = Path(cfile).stem
                for target_triple, filename_2_loop_metadata in target_triple_2_filename2df_loop_metadata.items():
                    if args.target_triple is not None and target_triple != args.target_triple:
                        continue

                    if not cfile in filename_2_loop_metadata:
                        print("Skipping because there are no loops")
                        continue
                    df_loop_data = filename_2_loop_metadata[cfile]

                    self.log_loop_metadata(
                        cfilename, df_loop_data, target_triple, config)

                    if len(df_loop_data) == 0:
                        print("Contains 0 loops: ", cfile)
                        continue

                    self.drop_uninteresting_data(df_loop_data)
                    self.pretty_print_found_loops(df_loop_data)

                    for num_unrolling in self.unrollings:
                        df_loop_data_resume = self.resume(
                            args, df_loop_data, cfilename, num_unrolling)
                        if self.start_loop is not None:
                            df_loop_data_resume.drop(
                                df_loop_data_resume[df_loop_data_resume['loopIdx'] < self.start_loop].index, inplace=True)
                        if self.end_loop is not None:
                            df_loop_data_resume.drop(
                                df_loop_data_resume[df_loop_data_resume['loopIdx'] > self.end_loop].index, inplace=True)
                        exec_time_dict = self.execute_and_validate_checksum(
                            df_loop_data_resume, num_unrolling, cfile, cfilename, target_triple, config)

        self.post_measurement()
        print("Done with application:", self.name)

    def resume(self, args, df_loop_data, filename, num_unrolling):
        if args.resume:
            if args.resumeFrom is not None:
                df_resume = pd.read_csv(args.resumeFrom, sep=";")
            else:
                resumeFrom = self.logs_dir + self.name + '/' + \
                    filename + '_' + str(num_unrolling) + '_times.txt'
                df_resume = pd.read_csv(resumeFrom, sep=";")
            df_resume = df_resume[::-1]
            for _, row in df_resume.iterrows():
                loop_idx = row['loopIdx']
                if loop_idx >= 0 and loop_idx <= len(df_loop_data):
                    resume_idx = (loop_idx + 1) % len(df_loop_data)
                    df_resume_head = df_loop_data.head(resume_idx)
                    df_resume_tail = df_loop_data.tail(
                        len(df_loop_data) - resume_idx)
                    df_loop_data = pd.concat([df_resume_tail, df_resume_head])
                    print(df_loop_data.to_string())
                    break
        return df_loop_data

    def drop_uninteresting_data(self, df_loop_data):
        df_loop_data.drop('startCol', axis=1, inplace=True)
        df_loop_data.drop('endCol', axis=1, inplace=True)
        df_loop_data.drop('startIsImplicitCode', axis=1, inplace=True)
        df_loop_data.drop('endIsImplicitCode', axis=1, inplace=True)


    def log_loop_metadata(self, cfile, df_loop_data, target_triple, config):
        loop_metadata_file_path = self.logs_dir + \
            self.name + '/' + cfile + '_' + target_triple + '_metadata.csv'
        loop_metadata_file = Path(loop_metadata_file_path)
        loop_metadata_file_exists = loop_metadata_file.is_file()
        if not loop_metadata_file_exists:
            df_loop_data.to_csv(loop_metadata_file_path, index=False)


    def parseArgs(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--unrolling', nargs="+", type=int,
                            help='Number of unrollings.')
        parser.add_argument('--logs_dir', type=str,
                            help='Directory in which log files (e.g. execution times) are written to.')
        parser.add_argument('--num_iterations', type=int,
                            help='Number of times each program should be executed.')
        parser.add_argument('--exclude_files', nargs="+", type=str,
                            help='Provide a list of source files. Loops in these source files will not be considered.')
        parser.add_argument('--include_files', nargs="+", type=str,
                            help='Provide a list of source files. Only loops in these files will be considered.')
        parser.add_argument(
            '--resumeFrom', type=str, help='Path to log file containing execution times. Measurements will continue where they stopped.')
        parser.add_argument('--resume', dest='resume', action='store_true')
        parser.add_argument('--start_loop', type=int,
                            help='Only compiles and executes loops starting from index n.')
        parser.add_argument('--end_loop', type=int,
                            help='Only compiles and executes loops ending at index n.')
        parser.add_argument('--target_triple', type=str,
                            help='Only compile modules matching target_triple for compilation.')
        parser.add_argument('--configs', nargs="+", type=str,
                            help='List of compile configurations to measure.')
        parser.add_argument('--filter_branchless', dest='filter_branchless',
                            action='store_true', help='Only measure loops that contain a branch.')
        parser.add_argument('--build-dir', type=str, help='Directory in which the build files will be located')
        parser.set_defaults(resume=False)

        args = parser.parse_args()
        if args.unrolling is not None:
            self.unrollings = args.unrolling
        if args.logs_dir is not None:
            self.logs_dir = args.logs_dir
        if args.num_iterations is not None:
            self.num_iterations = args.num_iterations
        if args.exclude_files is not None and args.include_files is not None:
            print('--exclude_files and --include_files cannot be used at the same time.')
            quit()
        if args.start_loop is not None:
            self.start_loop = args.start_loop
        if args.end_loop is not None:
            self.end_loop = args.end_loop
        if args.configs is not None:
            self.compile_configs = args.configs
        if args.filter_branchless is not None:
            self.filter_branchless = args.filter_branchless
        if args.build_dir is not None:
            self.build_dir = args.build_dir

        return args
