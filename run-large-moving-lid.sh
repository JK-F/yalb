N=10000
SIZE=300
lidv=0.3
omega=1.0

OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=8 ./build/executables/milestone05 --N=$N --size=$SIZE --lidv=$lidv --omega=$omega
python3 ./data/diff_frames.py $lidv $omega ./data/06_velocity.csv
