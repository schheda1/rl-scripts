import sys
import os
from executor import Executor


class ExecutorXSBenchCuda(Executor):
    def execute_command(self):
        return "./xsbench -s small -m event"

    def get_executable_name(self):
        return "./xsbench"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        if "(WARNING - INAVALID CHECKSUM!)" in output:
            return False
        return "Verification checksum: 945990 (Valid)" in output

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if "Runtime:" in line:
                runtime = float(line.split()[-2])
                return runtime
        print("Runtime not found")
        return -1

    def pre_measurement(self):
        source_dir = "../XSBench/cuda/"
        patch_makefile = "patches/xsbench-cuda-makefile"
        patch_header = "patches/XSbench_header.cuh"
        target_dir = self.build_dir + "xsbench-cuda/"
        target_makefile = self.build_dir + "xsbench-cuda/Makefile"
        target_header = self.build_dir + "xsbench-cuda/XSbench_header.cuh"

        self.exec(["cp", "-R", source_dir, target_dir], quit_at_error=True)
        self.exec(["cp", patch_makefile, target_makefile], quit_at_error=True)
        self.exec(["cp", patch_header, target_header], quit_at_error=True)
        os.chdir(self.build_dir + "xsbench-cuda")


if __name__ == "__main__":
    executor = ExecutorXSBenchCuda(
        "xsbench-cuda", include_files=["Simulation.cu"], end_loop=13
    )
    executor.execute()
