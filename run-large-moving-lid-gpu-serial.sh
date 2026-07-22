N=10000
SIZE=300
lidv=0.3
omega=1.0

# Requires build-cuda to already be configured/built: `make initgpu-turing && make allgpu`
./build-cuda/executables/milestone05 --N=$N --size=$SIZE --lidv=$lidv --omega=$omega
python3 ./data/diff_frames.py $lidv $omega
