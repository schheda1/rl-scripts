import os
from executor import Executor


class ExecutorcomplexCuda(Executor):
    def execute_command(self):
        return "./main 10000000 1000"

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
            if "Average kernel" in line:
                total_time += float(line.split()[-2])
        if total_time == 0:
            print("Could not find execution time in output: " + output)
            quit()
        return total_time

    def pre_measurement(self):
        source_dir = "../HeCBench/complex-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "complex-cuda/")


if __name__ == "__main__":
    executor = ExecutorcomplexCuda("complex-cuda", include_files=["main.cu"])
    executor.execute()
