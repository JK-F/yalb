all:
	cmake -B build
	cmake --build build

run:
	./build/executables/milestone02/milestone02 | sed  -e 's/,]/]/g'  -e 's/,}/}/g' > ./data/milestone02.json
