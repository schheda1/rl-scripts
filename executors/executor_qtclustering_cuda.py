import os
from executor import Executor


class ExecutorQtclusteringCuda(Executor):

    def execute_command(self):
        return './qtc'

    def get_executable_name(self):
        return './qtc'

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        return True

    def get_exec_time_from_output(self, output):
        total_time = 0
        for line in output.splitlines():
            if "qtc:" in line:
                split_line = line.split()
                total_time += float(split_line[-2])
        if total_time == 0:
            print("Could not find time in output")
            quit()
        return total_time
        
    def pre_measurement(self):
        source_dir = '../HeCBench/qtclustering-cuda/'
        self.exec(['cp', '-R', source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + 'qtclustering-cuda/')

if __name__ == "__main__":
    executor = ExecutorQtclusteringCuda('qtclustering-cuda', include_files=['QTC.cu'])
    executor.execute()
