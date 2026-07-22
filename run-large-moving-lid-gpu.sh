N=10000
SIZE=300
lidv=0.3
omega=1.0

# This machine has a single GPU, so every MPI rank would contend for the
# same device - keep NP=1 unless you're running on a multi-GPU box (in
# which case SIZE should divide evenly by sqrt(NP), see combine_csvs.py).
NP=1

# Requires build-cuda to already be configured/built: `make initgpu-turing && make allgpu`
mpirun -np $NP ./build-cuda/executables/milestone06 --N=$N --size_x=$SIZE --size_y=$SIZE --lidv=$lidv --omega=$omega
python3 ./data/combine_csvs.py --output ./data/06_velocity.csv
python3 ./data/diff_frames.py $lidv $omega ./data/06_velocity.csv
