DEFAULT_LOOP_IDX = 999999
HEURISTIC_LOOP_IDX = 888888


def int_list_to_string(int_list):
    return ",".join([str(x) for x in int_list])


def prettify_name(name, config, num_unrolling):
    if "uu-heuristic" in name:
        return "u&u-heuristic"
    if name == "default":
        return "baseline"

    if name.startswith("loop-"):
        prettified_named = "loop " + name[5:]
        if config == "uu":
            return (
                prettified_named
                + " (config='u&u' with unroll factor={num_unrolling})".format(
                    num_unrolling=num_unrolling
                )
            )
        elif config.startswith("unroll"):
            return (
                prettified_named
                + " (config='unroll' with unroll factor={num_unrolling})".format(
                    num_unrolling=num_unrolling
                )
            )
        elif config.startswith("unmerge"):
            return prettified_named + " (config='unmerge')"
        return prettified_named

    return name


def get_header_and_data(lines, keywords):
    header = ""
    data = []
    found_header = 0
    for line in lines:
        if found_header == 1 and line.strip().startswith("Time"):
            header = line
            found_header = 2
            continue

        if found_header == 2:
            found_header = 3
            continue

        if found_header == 3:
            if line.strip() == "":
                return (header, data)
            data.append(line)

        for keyword in keywords:
            if keyword in line:
                found_header = 1
                continue

    return (header, data)


def header_to_string(header):
    # header.split() looks like ['Time', '(%)', 'Total', 'Time', '(ns)', 'Instances', 'Avg', '(ns)', 'Med', '(ns)', 'Min', '(ns)', 'Max', '(ns)', 'StdDev', '(ns)', 'Name']
    # We want to split this into 2 lines:
    # line 1 : "Type","Time(%)","Time","Calls","Avg","Min","Max","Name"
    # line 2: ,%,s,,ms,ms,ms,
    split_header = header.split()
    first_line = '"Type","Time(%)","Time","Calls","Avg","Min","Max","Name"'

    type_value = ""
    time_percent = "%"
    total_time = split_header[4][1:-1]
    calls = ""
    avg_time_format = split_header[7][1:-1]
    min_time_format = split_header[11][1:-1]
    max_time_format = split_header[13][1:-1]
    name = ""
    second_line = (
        type_value
        + ","
        + time_percent
        + ","
        + total_time
        + ","
        + calls
        + ","
        + avg_time_format
        + ","
        + min_time_format
        + ","
        + max_time_format
        + ","
        + name
    )
    return (first_line, second_line)


def make_name_correct(name):
    return name.replace("Device-to-Host", "DtoH").replace("Host-to-Device", "HtoD")


def parse_data(data):
    name_begin_idx = -1
    for c in data:
        if c.isdigit() or c == "." or c == "," or c.isspace():
            name_begin_idx += 1
        else:
            break
    name = data[name_begin_idx:].strip()
    data = data[:name_begin_idx]
    split_data = data.split()

    # remove median
    split_data.pop(4)
    # remove StdDev
    split_data.pop(6)

    # replace all commas with nothing
    split_data = [x.replace(",", "") for x in split_data]
    name = make_name_correct(name)
    return (name, split_data)


def get_data_line(name, split_data):
    return '"GPU activities",' + ",".join(split_data) + ',"' + name + '"'


def contains_parseable_data(output):
    return ("cuda_gpu_kern_sum" in output or "gpukernsum" in output) and (
        "cuda_gpu_mem_time_sum" in output or "gpumemtimesum" in output
    )


def parse_nsys_output(lines):
    # this function takes the output of nsys profile and returns a csv string in the same format as nvprof's csv output
    kernel_exec_times = get_header_and_data(lines, ["cuda_gpu_kern_sum", "gpukernsum"])
    gpu_mem_times = get_header_and_data(
        lines, ["cuda_gpu_mem_time_sum", "gpumemtimesum"]
    )

    all_lines = []
    for data in kernel_exec_times[1]:
        name, split_data = parse_data(data)
        all_lines.append(get_data_line(name, split_data))
    for data in gpu_mem_times[1]:
        name, split_data = parse_data(data)
        all_lines.append(get_data_line(name, split_data))
    header = header_to_string(kernel_exec_times[0])
    return "\n".join(header) + "\n" + "\n".join(all_lines)
