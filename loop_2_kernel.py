def get_kernels(loop_idx, app_name):
    global app_2_loop_2_kernels
    # init if app_2_loop_2_kernels is empty
    if len(app_2_loop_2_kernels) == 0:
        init_kernel_map(app_2_loop_2_kernels)
    return app_2_loop_2_kernels[app_name]['loop-' + str(loop_idx)]

def measure_all_kernels(app_name):
    app_one_kernel = {'bezier-surface', 'bspline-vgh', 'clink', 'lavamd', 'mandelbrot', 'ccs', 'complex', 'haccmk',
                      'rainflow'}
    return app_name in app_one_kernel


def add_contract_kernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['contract'] = {}
    for i in range(1, 37):
        app_2_loop_2_kernels['contract']['loop-' + str(i)] = [
            'void contraction<float>(float const *, float const *, float*, int, int, int)']
    for i in range(39, 75):
        app_2_loop_2_kernels['contract']['loop-' + str(i)] = [
            'void contraction<double>(double const *, double const *, double*, int, int, int)']


def add_quicksort_kernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['quicksort'] = {}
    app_2_loop_2_kernels['quicksort']['loop-0'] = [
        'void gqsort_kernel<unsigned int>(unsigned int*, unsigned int*, block_record<unsigned int>*, parent_record*, work_record<unsigned int>*)']
    app_2_loop_2_kernels['quicksort']['loop-3'] = [
        'void gqsort_kernel<unsigned int>(unsigned int*, unsigned int*, block_record<unsigned int>*, parent_record*, work_record<unsigned int>*)']

    app_2_loop_2_kernels['quicksort']['loop-7'] = [
        'void lqsort_kernel<unsigned int>(unsigned int*, unsigned int*, work_record<unsigned int>*)']
    app_2_loop_2_kernels['quicksort']['loop-10'] = [
        'void lqsort_kernel<unsigned int>(unsigned int*, unsigned int*, work_record<unsigned int>*)']
    app_2_loop_2_kernels['quicksort']['loop-13'] = [
        'void lqsort_kernel<unsigned int>(unsigned int*, unsigned int*, work_record<unsigned int>*)']

    app_2_loop_2_kernels['quicksort']['loop-19'] = [
        'void gqsort_kernel<float>(float*, float*, block_record<float>*, parent_record*, work_record<float>*)']
    app_2_loop_2_kernels['quicksort']['loop-22'] = [
        'void gqsort_kernel<float>(float*, float*, block_record<float>*, parent_record*, work_record<float>*)']

    app_2_loop_2_kernels['quicksort']['loop-26'] = ['void lqsort_kernel<float>(float*, float*, work_record<float>*)']
    app_2_loop_2_kernels['quicksort']['loop-29'] = ['void lqsort_kernel<float>(float*, float*, work_record<float>*)']
    app_2_loop_2_kernels['quicksort']['loop-32'] = ['void lqsort_kernel<float>(float*, float*, work_record<float>*)']

    app_2_loop_2_kernels['quicksort']['loop-38'] = [
        'void gqsort_kernel<double>(double*, double*, block_record<double>*, parent_record*, work_record<double>*)']
    app_2_loop_2_kernels['quicksort']['loop-41'] = [
        'void gqsort_kernel<double>(double*, double*, block_record<double>*, parent_record*, work_record<double>*)']

    app_2_loop_2_kernels['quicksort']['loop-45'] = [
        'void lqsort_kernel<double>(double*, double*, work_record<double>*)']
    app_2_loop_2_kernels['quicksort']['loop-48'] = [
        'void lqsort_kernel<double>(double*, double*, work_record<double>*)']
    app_2_loop_2_kernels['quicksort']['loop-51'] = [
        'void lqsort_kernel<double>(double*, double*, work_record<double>*)']


def add_xsbench_kernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['xsbench'] = {}
    loops = [0, 1, 3, 6, 7, 8, 9, 10, 11, 12, 13]
    for i in loops:
        app_2_loop_2_kernels['xsbench']['loop-' + str(i)] = [
            'xs_lookup_kernel_baseline(Inputs, SimulationData)']
    for i in range(14, 800):
        app_2_loop_2_kernels['xsbench']['loop-' + str(i)] = [
            'xs_lookup_kernel_baseline(Inputs, SimulationData)']


def add_bn_kernel(app_2_loop_2_kernels):
    app_2_loop_2_kernels['bn'] = {}
    genScoreKernelLoops = [0, 1, 5, 7, 8]
    computeScoreKernelLoops = [15, 16, 19, 21, 25]
    genScoreKernelName = 'genScoreKernel(int, float*, int const *, float const *)'
    computeKernelName = 'computeKernel(int, int, float const *, bool const *, int, int, float*, int*)'
    for i in genScoreKernelLoops:
        app_2_loop_2_kernels['bn']['loop-' + str(i)] = [genScoreKernelName]
    for i in computeScoreKernelLoops:
        app_2_loop_2_kernels['bn']['loop-' + str(i)] = [computeKernelName]

    # 11 is used in both kernels
    app_2_loop_2_kernels['bn']['loop-11'] = [genScoreKernelName, computeKernelName]


def addCoordinatesKernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['coordinates'] = {}
    loop1_kernel = 'void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<cartesian_2d<double>>, cartesian_2d<double>>, unsigned long>, thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<cartesian_2d<double>>, cartesian_2d<double>>, unsigned long>(cartesian_2d<double>, thrust::device_ptr<cartesian_2d<double>>)'
    loop7_kernel = 'void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<cartesian_2d<float>>, cartesian_2d<float>>, unsigned long>, thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<cartesian_2d<float>>, cartesian_2d<float>>, unsigned long>(cartesian_2d<float>, thrust::device_ptr<cartesian_2d<float>>)'
    app_2_loop_2_kernels['coordinates']['loop-1'] = [loop1_kernel]
    app_2_loop_2_kernels['coordinates']['loop-7'] = [loop7_kernel]
    loop_3_5_kernel = 'void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__transform::unary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<lonlat_2d<double> const >>, thrust::detail::normal_iterator<thrust::device_ptr<cartesian_2d<double>>>, thrust::cuda_cub::__transform::no_stencil_tag, to_cartesian_functor<double>, thrust::cuda_cub::__transform::always_true_predicate>, long>, thrust::cuda_cub::__transform::unary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<lonlat_2d<double> const >>, thrust::detail::normal_iterator<thrust::device_ptr<cartesian_2d<double>>>, thrust::cuda_cub::__transform::no_stencil_tag, to_cartesian_functor<double>, thrust::cuda_cub::__transform::always_true_predicate>, long>(lonlat_2d<double> const , thrust::device_ptr<lonlat_2d<double> const >)'
    loop_9_11_kernel = 'void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__transform::unary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<lonlat_2d<float> const >>, thrust::detail::normal_iterator<thrust::device_ptr<cartesian_2d<float>>>, thrust::cuda_cub::__transform::no_stencil_tag, to_cartesian_functor<float>, thrust::cuda_cub::__transform::always_true_predicate>, long>, thrust::cuda_cub::__transform::unary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<lonlat_2d<float> const >>, thrust::detail::normal_iterator<thrust::device_ptr<cartesian_2d<float>>>, thrust::cuda_cub::__transform::no_stencil_tag, to_cartesian_functor<float>, thrust::cuda_cub::__transform::always_true_predicate>, long>(lonlat_2d<float> const , thrust::device_ptr<lonlat_2d<float> const >)'
    app_2_loop_2_kernels['coordinates']['loop-3'] = [loop_3_5_kernel]
    app_2_loop_2_kernels['coordinates']['loop-5'] = [loop_3_5_kernel]
    app_2_loop_2_kernels['coordinates']['loop-9'] = [loop_9_11_kernel]
    app_2_loop_2_kernels['coordinates']['loop-11'] = [loop_9_11_kernel]
    app_2_loop_2_kernels['coordinates']['loop-12'] = [loop1_kernel, loop7_kernel]


def add_libor_kernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['libor'] = {}
    gpuKernelLoops = [4, 6, 9, 10, 18]
    gpu2KernelLoops = [1, 15, 20]
    gpuKernelName = 'Pathcalc_Portfolio_KernelGPU(float*, float*, float const *, int const *, float const *, float, int, int, int)'
    gpu2KernelName = 'Pathcalc_Portfolio_KernelGPU2(float*, float const *, int const *, float const *, float, int, int, int)'
    for i in gpuKernelLoops:
        app_2_loop_2_kernels['libor']['loop-' + str(i)] = [gpuKernelName]
    for i in gpu2KernelLoops:
        app_2_loop_2_kernels['libor']['loop-' + str(i)] = [gpu2KernelName]


def add_qtclustering_kernels(app_2_loop_2_kernels):
    app_2_loop_2_kernels['qtclustering'] = {}
    app_2_loop_2_kernels['qtclustering']['loop-0'] = ['reduce_card_device(int*, int)']

    computeDegreesKernel = 'compute_degrees(int*, int*, int, int)'
    app_2_loop_2_kernels['qtclustering']['loop-1'] = [computeDegreesKernel]
    app_2_loop_2_kernels['qtclustering']['loop-2'] = [computeDegreesKernel]

    trimKernel = 'trim_ungrouped_pnts_indr_array(int, int*, float*, int*, char*, char*, int*, int*, float*, int*, int, int, int, float)'
    app_2_loop_2_kernels['qtclustering']['loop-4'] = [trimKernel]

    qtcKernel = 'QTC_device(float*, char*, char*, int*, int*, int*, float*, int*, int, int, int, float, int, int, int)'

    # 8 to 21 are qtc and trim kernels
    for i in range(8, 22):
        app_2_loop_2_kernels['qtclustering']['loop-' + str(i)] = [qtcKernel, trimKernel]
    app_2_loop_2_kernels['qtclustering']['loop-24'] = [qtcKernel, trimKernel]


def init_kernel_map(app_2_loop_2_kernels):
    add_contract_kernels(app_2_loop_2_kernels)
    add_quicksort_kernels(app_2_loop_2_kernels)
    add_xsbench_kernels(app_2_loop_2_kernels)
    add_bn_kernel(app_2_loop_2_kernels)
    addCoordinatesKernels(app_2_loop_2_kernels)
    add_libor_kernels(app_2_loop_2_kernels)
    add_qtclustering_kernels(app_2_loop_2_kernels)

def kernel_name_2_pretty(kernel_name):
    if kernel_name.startswith('bspline'):
        return 'bspline'
    if kernel_name.startswith('compute_bicluster'):
        return 'compute_bicluster'
    if kernel_name.startswith('lstm_inference'):
        return 'lstm_inference'
    if kernel_name.startswith('void contraction<double>'):
        return 'contraction<double>'
    if kernel_name.startswith('void contraction<float>'):
        return 'contraction<float>'
    if "lonlat_2d<double>" in kernel_name:
        return "cub::lonlat_2d<double>"
    if "lonlat_2d<float>" in kernel_name:
        return "cub::lonlat_2d<float>"
    if "uninitialized_fill" in kernel_name and "cartesian_2d<double>" in kernel_name:
        return "cub::uninitialized_fill<double>"
    if "uninitialized_fill" in kernel_name and "cartesian_2d<float>" in kernel_name:
        return "cub::uninitialized_fill<float>"
    if kernel_name.startswith('haccmk_kernel'):
        return 'haccmk_kernel'
    if kernel_name.startswith('md(box_str'):
        return 'md'
    if kernel_name.startswith('Pathcalc_Portfolio_KernelGPU2'):
        return 'Pathcalc_Portfolio_KernelGPU2'
    if kernel_name.startswith('Pathcalc_Portfolio_KernelGPU'):
        return 'Pathcalc_Portfolio_KernelGPU'
    if kernel_name.startswith('mandel'):
        return 'mandel'
    if kernel_name.startswith('QTC_device'):
        return 'QTC_device'
    if kernel_name.startswith('reduce_card_device'):
        return 'reduce_card_device'
    if kernel_name.startswith('update_clustered_pnts_mask'):
        return 'update_clustered_pnts_mask'
    if kernel_name.startswith('trim_ungrouped_pnts_indr_array'):
        return 'trim_ungrouped_pnts_indr_array'
    if kernel_name.startswith('compute_degrees'):
        return 'compute_degrees'
    if kernel_name.startswith('void gqsort_kernel<double>'):
        return 'gqsort_kernel<double>'
    if kernel_name.startswith('void gqsort_kernel<float>'):
        return 'gqsort_kernel<float>'
    if kernel_name.startswith('void gqsort_kernel<unsigned'):
        return 'gqsort_kernel<unsigned>'
    if kernel_name.startswith('void lqsort_kernel<double>'):
        return 'lqsort_kernel<double>'
    if kernel_name.startswith('void lqsort_kernel<float>'):
        return 'lqsort_kernel<float>'
    if kernel_name.startswith('void lqsort_kernel<unsigned'):
        return 'lqsort_kernel<unsigned>'
    if kernel_name.startswith('rainflow_count'):
        return 'rainflow_count'
    if kernel_name.startswith('xs_lookup_kernel_baseline'):
        return 'xs_lookup_kernel_baseline'
    if "DeviceReduceKernel" in kernel_name:
        return "cub::DeviceReduceKernel"
    if "DeviceReduceSingleTileKernel" in kernel_name:
        return "cub::DeviceReduceSingleTileKernel"

    kernel_2_pretty = {}
    kernel_2_pretty['genScoreKernel(int, float*, int const *, float const *)'] = 'genScoreKernel'
    kernel_2_pretty['computeKernel(int, int, float const *, bool const *, int, int, float*, int*)'] = 'computeKernel'
    kernel_2_pretty['BezierGPU(XYZ const *, XYZ*, int, int, int, int)'] = 'BezierGPU'
    kernel_2_pretty['complex_double(char*, int)'] = 'complex_double'
    kernel_2_pretty['complex_float(char*, int)'] = 'complex_float'
    if kernel_name in kernel_2_pretty:
        return kernel_2_pretty[kernel_name]
    return kernel_name