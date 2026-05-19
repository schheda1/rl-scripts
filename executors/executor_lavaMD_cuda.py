import os
from executor import Executor


class ExecutorLavaMDCuda(Executor):
    def execute_command(self):
        return "./main -boxes1d 30"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        return True

    def get_exec_time_from_output(self, output):
        parse_next_line = False
        for line in output.splitlines():
            if parse_next_line:
                split_line = line.split()
                time = float(split_line[0])
                return time
            if "Kernel execution time" in line:
                parse_next_line = True
        print("Could not find time in output")
        quit()

    def pre_measurement(self):
        source_dir = "../HeCBench/lavaMD-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        self.exec(["cp", "-R", "../HeCBench/data", self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "lavaMD-cuda/")


if __name__ == "__main__":
    executor = ExecutorLavaMDCuda("lavamd-cuda", include_files=["main.cu"])
    executor.execute()
