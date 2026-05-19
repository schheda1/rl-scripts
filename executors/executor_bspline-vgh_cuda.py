import os
from executor import Executor


class ExecutorbsplinevghCuda(Executor):
    def execute_command(self):
        return "./main"

    def get_executable_name(self):
        return "./main"

    def make_command(self, additional_cflags=""):
        make_command = 'make EXTRA_CFLAGS="{additional_cflags}"'
        return make_command.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        # looking for line 'walkers[0]->collect([resVal resGrad resHess]) = [-1.315770e+04 1.767734e+04 3.817090e+05]'

        search_string = "walkers[0]->collect([resVal resGrad resHess]) = "
        search_string_len = len(search_string)
        for line in output.splitlines():
            if line.startswith(search_string):
                line = line[search_string_len:]
                line = line.replace("[", "")
                line = line.replace("]", "")
                line_split = line.split()
                if len(line_split) != 3:
                    return False

                numbers = [float(x) for x in line_split]

                def eps_eq(a, b, eps=1e-3):
                    return abs(a - b) < eps * abs(b)

                if not eps_eq(numbers[0], -1.315770e04):
                    return False
                if not eps_eq(numbers[1], 1.767734e04):
                    return False
                if not eps_eq(numbers[2], 3.817090e05):
                    return False
                return True

        return False

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if "Total kernel execution" in line:
                return float(line.split()[-2])
        print("time not found")
        quit()

    def pre_measurement(self):
        source_dir = "../HeCBench/bspline-vgh-cuda/"
        self.exec(["cp", "-R", source_dir, self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + "bspline-vgh-cuda/")


if __name__ == "__main__":
    executor = ExecutorbsplinevghCuda("bspline-vgh-cuda", include_files=["main.cu"])
    executor.execute()
