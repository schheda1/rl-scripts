import os
from executor import Executor


class ExecutorccsCuda(Executor):
    def execute_command(self):
        return "./main -t 0.9 -i Data_Constant_100_1_bicluster.txt -o ./Output.txt -m 50 -p 1 -g 100.0 -r 100"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        def file_2_str(file_name):
            with open(file_name, "r") as file:
                return "".join(line.strip() for line in file)

        def files_are_equal(file1, file2):
            file1_string = file_2_str(file1)
            file2_string = file_2_str(file2)
            return file1_string == file2_string

        # Checks whether Output.txt is equal to ccs_constant_100_1.out (replaces the diff -Bb command in HeCBench)
        return files_are_equal("Output.txt", "ccs_constant_100_1.out")

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if "Average kernel" in line:
                return float(line.split()[-2])
        print("time not found")
        quit()

    def pre_measurement(self):
        source_dir = "../HeCBench/ccs-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "ccs-cuda/")


if __name__ == "__main__":
    executor = ExecutorccsCuda("ccs-cuda", include_files=["main.cu"])
    executor.execute()
