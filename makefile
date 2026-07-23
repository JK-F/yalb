all:
	cmake --build build -j $(nproc)
init:
	cmake -B build

initgpu:
	cmake -DCMAKE_BUILD_TYPE=Release \
	      -DKokkos_ARCH_PASCAL61=ON \
	      -DKokkos_ENABLE_CUDA=ON \
	      -DCMAKE_CUDA_COMPILER=nvcc \
	      -DKokkos_ENABLE_CUDA_LAMBDA=ON \
	      -B build-cuda
run2:
	./build/executables/milestone02
run3:
	./build/executables/milestone03
run4:
	./build/executables/milestone04
run5:
	./build/executables/milestone05

run5p:
	OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=8 ./build/executables/milestone05 --size=300 --lidv=0.3 --omega=1.0 --N=10000

# GPU build for this machine's local GPU (Turing, e.g. GTX 1660 Ti). Needs
# `build/` already configured once (via `init`) so Kokkos' nvcc_wrapper is
# available, and the `cuda` package (nvcc) + a compatible host compiler
# installed (g++-15 on this machine, since the default GCC is too new for nvcc).
initgpu-turing: init
	PATH="/opt/cuda/bin:$$PATH" NVCC_WRAPPER_DEFAULT_COMPILER=g++-15 \
	cmake -DCMAKE_CXX_COMPILER=$(CURDIR)/build/_deps/kokkos-src/bin/nvcc_wrapper \
	      -DYALB_ENABLE_CUDA=ON \
	      -DKokkos_ARCH_TURING75=ON \
	      -B build-cuda

allgpu:
	PATH="/opt/cuda/bin:$$PATH" cmake --build build-cuda -j $(nproc)

run6gpu:
	mpirun -np 1 ./build-cuda/executables/milestone06 --size_x=300 --size_y=300 --lidv=0.3 --omega=1.0 --N=10000

run5gpu:
	./build-cuda/executables/milestone05 --size=300 --lidv=0.3 --omega=1.0 --N=10000


