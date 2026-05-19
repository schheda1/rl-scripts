import os
from executor import Executor


class ExecutormandelbrotCuda(Executor):
    def execute_command(self):
        return "./mandelbrot 100"

    def get_executable_name(self):
        return "./mandelbrot"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        for line in output.splitlines():
            if "Success" in line:
                return True
        return False

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if "Average kernel execution" in line:
                split_line = line.split()
                time = float(split_line[-2])
                return time
        print("Could not find time in output")
        quit()

    def pre_measurement(self):
        source_dir = "../HeCBench/mandelbrot-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "mandelbrot-cuda/")


if __name__ == "__main__":
    executor = ExecutormandelbrotCuda("mandelbrot-cuda", include_files=["main.cu"])
    executor.execute()
