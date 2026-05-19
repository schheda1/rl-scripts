import os
from executor import Executor


class ExecutorHaccmkCuda(Executor):
    def execute_command(self):
        return "./haccmk 2000"

    def get_executable_name(self):
        return "./haccmk"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        for line in output.splitlines():
            if "PASS" in line:
                return True
        return False

    def get_exec_time_from_output(self, output):
        return -1

    def pre_measurement(self):
        source_dir = "../HeCBench/haccmk-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "haccmk-cuda/")


if __name__ == "__main__":
    executor = ExecutorHaccmkCuda("haccmk-cuda", include_files=["haccmk.cu"])
    executor.execute()
