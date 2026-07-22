#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include "mpi_util.hpp"
#include <cstdio>
#include <string>
#include <sys/types.h>
#include <mpi.h>



#define DEFAULT_PRINT false
#define PRINT_STEADY true
#define DEFAULT_TIMESTEPS 2000
#define PRINT_TIMESTEP(i, timesteps) printf("Bolzman Lattice Timestep %05d / %05d", i, timesteps)
#define PRINT_MLUPS(runtime, X, Y, N) printf("Done computing after %fs\nMLUPS: %f\n", runtime, static_cast<double>(X * Y * N) / (runtime * 1000000.) )

#define DEFAULT_SIZE 30
#define SIZE_X DEFAULT_SIZE
#define SIZE_Y DEFAULT_SIZE
#define GHOST_BUFFERS 1

#define DENSITY_RHO 1
#define INIT_SPEED 0
#define DEFAULT_LIDV 0.1
#define DEFAULT_OMEGA 1.5

// Exchanges the 1-cell ghost layer of `simulation.distribution` with the
// four Cartesian neighbors. X is exchanged before Y and each message spans
// the full column/row (including the corner cells); since every rank runs
// the two phases in the same order, the Y phase forwards the corner data
// the X phase just received, so diagonal ghost corners end up correct too.
//
// Boundary slices are packed into freshly-allocated (hence always
// contiguous) staging buffers via a kernel rather than deep_copy'd directly
// out of a Kokkos::subview: fixing one index of `distrib` gives a strided,
// non-contiguous slice under LayoutLeft (Cuda's default View layout), and
// Kokkos can't deep_copy a non-contiguous view across memory spaces that
// share no common execution space (e.g. Cuda <-> Host).
void exchange_ghost_layer(MPI_Comm &cart, int rank, BoltzmanLattice &simulation) {
  auto distrib = simulation.distribution;
  int sx = static_cast<int>(simulation.size_x);
  int sy = static_cast<int>(simulation.size_y);
  using Boundary = Kokkos::View<double**>;
  {
    int left  = neighbors_rank(rank, cart, LEFT);
    int right = neighbors_rank(rank, cart, RIGHT);

    Boundary send_l("send_l", sy, NUM_DIRECTIONS), send_r("send_r", sy, NUM_DIRECTIONS);
    Boundary recv_l("recv_l", sy, NUM_DIRECTIONS), recv_r("recv_r", sy, NUM_DIRECTIONS);

    Kokkos::parallel_for("pack_x_ghost", Kokkos::RangePolicy<>(0, sy), KOKKOS_LAMBDA(const int y) {
      for (int d = 0; d < NUM_DIRECTIONS; d++) {
        send_l(y, d) = distrib(1,      y, d);
        send_r(y, d) = distrib(sx - 2, y, d);
      }
    });

    auto send_l_h = Kokkos::create_mirror_view(send_l);
    auto send_r_h = Kokkos::create_mirror_view(send_r);
    auto recv_l_h = Kokkos::create_mirror_view(recv_l);
    auto recv_r_h = Kokkos::create_mirror_view(recv_r);

    Kokkos::deep_copy(send_l_h, send_l);
    Kokkos::deep_copy(send_r_h, send_r);

    int count = sy * NUM_DIRECTIONS;
    MPI_Sendrecv(send_l_h.data(), count, MPI_DOUBLE, left,  0,
                 recv_r_h.data(), count, MPI_DOUBLE, right, 0,
                 cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_r_h.data(), count, MPI_DOUBLE, right, 1,
                 recv_l_h.data(), count, MPI_DOUBLE, left,  1,
                 cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_l, recv_l_h);
    Kokkos::deep_copy(recv_r, recv_r_h);

    Kokkos::parallel_for("unpack_x_ghost", Kokkos::RangePolicy<>(0, sy), KOKKOS_LAMBDA(const int y) {
      for (int d = 0; d < NUM_DIRECTIONS; d++) {
        distrib(0,      y, d) = recv_l(y, d);
        distrib(sx - 1, y, d) = recv_r(y, d);
      }
    });
  }

  {
    int up   = neighbors_rank(rank, cart, UP);
    int down = neighbors_rank(rank, cart, DOWN);

    Boundary send_u("send_u", sx, NUM_DIRECTIONS), send_d("send_d", sx, NUM_DIRECTIONS);
    Boundary recv_u("recv_u", sx, NUM_DIRECTIONS), recv_d("recv_d", sx, NUM_DIRECTIONS);

    Kokkos::parallel_for("pack_y_ghost", Kokkos::RangePolicy<>(0, sx), KOKKOS_LAMBDA(const int x) {
      for (int d = 0; d < NUM_DIRECTIONS; d++) {
        send_u(x, d) = distrib(x, sy - 2, d);
        send_d(x, d) = distrib(x, 1,      d);
      }
    });

    auto send_u_h = Kokkos::create_mirror_view(send_u);
    auto send_d_h = Kokkos::create_mirror_view(send_d);
    auto recv_u_h = Kokkos::create_mirror_view(recv_u);
    auto recv_d_h = Kokkos::create_mirror_view(recv_d);

    Kokkos::deep_copy(send_u_h, send_u);
    Kokkos::deep_copy(send_d_h, send_d);

    int count = sx * NUM_DIRECTIONS;
    MPI_Sendrecv(send_u_h.data(), count, MPI_DOUBLE, up,   2,
                 recv_d_h.data(), count, MPI_DOUBLE, down, 2,
                 cart, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_d_h.data(), count, MPI_DOUBLE, down, 3,
                 recv_u_h.data(), count, MPI_DOUBLE, up,   3,
                 cart, MPI_STATUS_IGNORE);

    Kokkos::deep_copy(recv_u, recv_u_h);
    Kokkos::deep_copy(recv_d, recv_d_h);

    Kokkos::parallel_for("unpack_y_ghost", Kokkos::RangePolicy<>(0, sx), KOKKOS_LAMBDA(const int x) {
      for (int d = 0; d < NUM_DIRECTIONS; d++) {
        distrib(x, sy - 1, d) = recv_u(x, d);
        distrib(x, 0,      d) = recv_d(x, d);
      }
    });
  }
}

void run_sublattice_simulation(MPI_Comm &cart, const uint &size_x, const uint &size_y, const double &omega, const double &lidv ,const uint &timesteps, const bool &print, const bool &is_root) {
  int rank;
  MPI_Comm_rank(cart, &rank);

  int coords[2], cart_dims[2], cart_periods[2];
  MPI_Cart_get(cart, 2, cart_dims, cart_periods, coords);

  BoltzmanLattice simulation(size_x, size_y, GHOST_BUFFERS, lidv, omega, INIT_SPEED, INIT_SPEED, DENSITY_RHO);
  simulation.open_files("./data/06_p" + std::to_string(coords[0]) + "_" + std::to_string(coords[1]));

  // Init ghost buffers to 0;
  // Prolly useless, since streaming comes before, but yeah, wont take long anyways
  simulation.init_ghost_buffers(0);
  simulation.feq_distrib();

  // print initial
  if (print)
    simulation.print_velocity(0);

  if (is_root) PRINT_TIMESTEP(0, timesteps);
  for (int i = 1; i <= timesteps; ++i) {
    if (is_root && i % 100 == 0) {
      printf("\r");
      PRINT_TIMESTEP(i, timesteps);
    }

    simulation.streaming();
    simulation.bounce_back();
    simulation.collision_fused();

    // The four neighbors; with periodic boundaries every rank has all four
    exchange_ghost_layer(cart, rank, simulation);

    if (print && i % 20 == 0)
      simulation.print_velocity(i);
  }
  if (is_root) printf("\n");

  if (PRINT_STEADY && !print) {
      simulation.print_velocity(timesteps);
  }
  if (PRINT_STEADY) {
      simulation.streaming();
      simulation.bounce_back();
      simulation.collision_fused();
      simulation.print_velocity(timesteps + 1);
  }
}

int main(int argc, char* argv[]) {
  double omega              = DEFAULT_OMEGA;
  double lid_velocity       = DEFAULT_LIDV;
  uint size_x               = DEFAULT_SIZE;
  uint size_y               = DEFAULT_SIZE;
  uint timesteps            = DEFAULT_TIMESTEPS;
  bool print                = DEFAULT_PRINT;
  for (int i = 0; i < argc; i++) {
    auto arg = std::string(argv[i]);
    if (arg.rfind("--omega=", 0) == 0) {
      omega = std::stod(arg.substr(8));
    }
    if (arg.rfind("--lidv=", 0) == 0) {
      lid_velocity = std::stod(arg.substr(7));
    }
    if (arg.rfind("--size_x=", 0) == 0) {
      size_x = std::stoi(arg.substr(9));
    }
    if (arg.rfind("--size_y=", 0) == 0) {
      size_y = std::stoi(arg.substr(9));
    }
    if (arg.rfind("--N=", 0) == 0) {
      timesteps = std::stoi(arg.substr(4));
    }
    if (arg.rfind("--print", 0) == 0) {
      print = true;
    }
  }
  MPI_Init(&argc, &argv);
  Kokkos::initialize(argc, argv);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  bool is_root = world_rank == 0;

  if (is_root) {
    printf("Omega: %f, Lid Velocity: %f, L: %d x %d, N: %d\n", omega, lid_velocity, size_x, size_y, timesteps);
    printf("Reynolds Number: %f\n", (lid_velocity * size_x) / ( (1/omega - 0.5)/3));
  }
  Kokkos::Timer timer;
  Kokkos::fence();

  int mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  int dims[2] = {0, 0}; // Number of nodes/processes in each dimension, will be filled by MPI
  MPI_Dims_create(mpi_size, 2, dims);

  MPI_Comm cart;
  int periods[2] = {0, 0}; // No periodic bounds in any dimension
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, /* reorder */ 1, &cart);


  int sublattice_size_x =  size_x / dims[0];
  int sublattice_size_y =  size_y / dims[1];
  run_sublattice_simulation(cart, sublattice_size_x, sublattice_size_y, omega, lid_velocity, timesteps, print, is_root);

  double local_runtime = timer.seconds();
  double runtime;
  MPI_Reduce(&local_runtime, &runtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (is_root) PRINT_MLUPS(runtime, size_x, size_y, timesteps);

  Kokkos::finalize();
  MPI_Finalize();

  return 0;
}
