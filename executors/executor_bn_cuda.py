import os
from executor import Executor


class ExecutorbnCuda(Executor):
    def execute_command(self):
        return "./main result"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        return True

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if "Kernel execution time" in line:
                return float(line.split()[-2])
        print("bn: time not found")
        quit()

    def pre_measurement(self):
        source_dir = "../HeCBench/bn-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "bn-cuda/")


if __name__ == "__main__":
    executor = ExecutorbnCuda("bn-cuda", include_files=["main.cu"])
    executor.execute()
