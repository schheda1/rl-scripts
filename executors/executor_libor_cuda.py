import os
from executor import Executor


class ExecutorliborCuda(Executor):
    def execute_command(self):
        return "./main 100"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        for line in output.splitlines():
            if "PASS" in line:
                return True
        return False

    def get_exec_time_from_output(self, output):
        total_time = 0
        for line in output.splitlines():
            if "Average kernel execution" in line:
                split_line = line.split()
                time = float(split_line[-2])
                return time
        if total_time == 0:
            print("Could not find time in output")
            quit()
        return total_time

    def pre_measurement(self):
        source_dir = "../HeCBench/libor-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "libor-cuda/")


if __name__ == "__main__":
    executor = ExecutorliborCuda("libor-cuda", include_files=["main.cu"])
    executor.execute()
