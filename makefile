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
	      -B build
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


