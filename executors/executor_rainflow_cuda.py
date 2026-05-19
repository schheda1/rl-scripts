import os
from executor import Executor

# Create a new class that inherits from the Executor class
class ExecutorrainflowCuda(Executor):
    def execute_command(self):
        # This is the command that will be executed to run the benchmark
        return "./main 100000 100"

    def get_executable_name(self):
        # This is the name of the executable file that is created by the build process
        return "./main"

    def make_command(self, additional_cflags=""):
        # This is the command that will be executed to build the benchmark
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        # This function is used to validate the output of the benchmark
        for line in output.splitlines():
            if "PASS" in line:
                return True
        return False

    def get_exec_time_from_output(self, output):
        # This function is used to extract the execution time of the benchmark from the output
        # This is not mandatory, as we use nvprof/nsys to measure the kernel execution times
        for line in output.splitlines():
            if "Average kernel execution" in line:
                split_line = line.split()
                time = float(split_line[-2])
                return time
        print("Could not find time in output")
        quit()

    def pre_measurement(self):
        # This function is used to prepare the benchmark for measurement.
        # In this case, we copy the benchmark source code to the build directory.
        # One could also clone the benchmark from a git repository or download it from a website instead.
        # In the end, one has to change the current working directory to the build directory.
        source_dir = "../HeCBench/rainflow-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "rainflow-cuda/")


if __name__ == "__main__":
    # The first argument is the name of the benchmark
    # Through the include_files argument, you can specify which files should be included in the measurement.
    # This is necessary since we only want to apply our pass to loops in the kernel code and not to the whole benchmark.
    executor = ExecutorrainflowCuda("rainflow-cuda", include_files=["main.cu"])
    executor.execute()
