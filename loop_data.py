class LoopData:
    def __init__(self):
        self.execution_times = []
        self.compilation_times = []
        self.loop_idx = -1
        self.code_size = 0

    def add_execution_time(self, execution_time):
        self.execution_times.append(execution_time)

    def add_compilation_time(self, compilation_time):
        self.compilation_times.append(compilation_time)
