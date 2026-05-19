import os
from executor import Executor


class ExecutorquicksortCuda(Executor):
    def execute_command(self):
        return "./main 10 2048 2048"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        pass_count = 0
        for line in output.splitlines():
            if "Number of failures: 0 out of 10" in line:
                pass_count += 1
        return pass_count == 3

    def get_exec_time_from_output(self, output):
        total_time = 0
        for line in output.splitlines():
            if "Average Time:" in line:
                split_line = line.split()
                time = float(split_line[-2])
                total_time += time
        if total_time == 0:
            print("Could not find time in output")
            quit()
        return total_time

    def pre_measurement(self):
        source_dir = "../HeCBench/quicksort-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "quicksort-cuda/")


if __name__ == "__main__":
    executor = ExecutorquicksortCuda("quicksort-cuda", include_files=["main.cu"])
    executor.execute()
