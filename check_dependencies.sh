function check_command() {
    if ! command -v $1 &> /dev/null
    then
        echo "$1 $2"
    else
        echo "$1 found"
    fi
}

check_command python3 "not found. Please install it."
check_command pip3 "not found. Please install it."
check_command cmake "not found. Please install it."
check_command make "not found. Please install it."
check_command ninja "not found. Please install it."

echo -e "\nChecking whether gcc or clang are installed. You need a C/C++ compiler to build LLVM."
check_command gcc "not found. You need a C/C++ compiler to build our fork of LLVM."
check_command clang "not found. You need a C/C++ compiler to build our fork of LLVM."

echo -e "\nChecking whether nvprof or nsys are installed. You need at least one of them."
echo -e "If your GPU has compute capability 8.0 or higher, you need to have nsys installed because nvprof won't work."
echo -e "You might have to add the CUDA bin directory to your PATH variable."
echo -e "For example: export PATH=\$PATH:/usr/local/cuda-12.3/bin"
check_command nvprof "not found. Install it if nsys is not available."
check_command nsys "not found. Install it if nvprof is not available or if your GPU has compute capability 8.0 or higher."

echo -e "\nChecking whether CUDA_HOME is set."
if [ -z "$CUDA_HOME" ]
then
    echo "CUDA_HOME is not set. Please set it to the path of your CUDA installation."
    echo "For example: export CUDA_HOME=/usr/local/cuda-12.3"
    echo "If nvprof or are nsys are found, you can run which nvprof or which nsys to find the path."
else
    echo "CUDA_HOME is set to $CUDA_HOME"
fi
