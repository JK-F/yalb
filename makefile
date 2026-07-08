all:
	cmake --build build
init:
	cmake -B build
run2:
	./build/executables/milestone02
run3:
	./build/executables/milestone03
run4:
	./build/executables/milestone04
run5:
	./build/executables/milestone05
run5p:
	OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=8 ./build/executables/milestone05
