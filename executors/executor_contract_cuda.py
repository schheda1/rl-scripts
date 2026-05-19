import os
from executor import Executor


class ExecutorcontractCuda(Executor):
    def execute_command(self):
        return "./main 64 5"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        checksum_count = 0
        for line in output.splitlines():
            if "Checksum: 99670818816.000000 min:64.000000 max:262144.000000" in line:
                checksum_count += 1
        return checksum_count == 2

    def get_exec_time_from_output(self, output):
        total_time = 0
        for line in output.splitlines():
            if "Average kernel execution time" in line:
                total_time += float(line.split()[-2])
        if total_time == 0:
            print("Could not find execution time in output: " + output)
            quit()
        return total_time

    def pre_measurement(self):
        source_dir = "../HeCBench/contract-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "contract-cuda/")


if __name__ == "__main__":
    executor = ExecutorcontractCuda("contract-cuda", include_files=["main.cu"])
    executor.execute()
