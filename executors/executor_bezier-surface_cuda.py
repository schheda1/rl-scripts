import os
from executor import Executor


class ExecutorbeziersurfaceCuda(Executor):

    def execute_command(self):
        return './bs -n 4096'

    def get_executable_name(self):
        return './bs'

    def make_command(self, additional_cflags=''):
        return 'make EXTRA_CFLAGS="{additional_cflags}"'.format(additional_cflags=additional_cflags)

    def validate_output(self, output):
        for line in output.splitlines():
            if 'PASS' in line:
                return True
        return False

    def get_exec_time_from_output(self, output):
        for line in output.splitlines():
            if 'kernel execution' in line:
                return float(line.split()[-2])
        print('Could not find execution time in output: ' + output)
        quit()

    def pre_measurement(self):
        self.exec(['cp', '-R', '../HeCBench/bezier-surface-cuda/', self.build_dir], quit_at_error=True)
        os.chdir(self.build_dir + 'bezier-surface-cuda/')

if __name__ == "__main__":
    executor = ExecutorbeziersurfaceCuda('bezier-surface-cuda', include_files=['main.cu'])
    executor.execute()
