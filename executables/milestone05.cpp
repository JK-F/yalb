#include <Kokkos_Core.hpp>
#include "Kokkos_Complex.hpp"
#include "boltzman.hpp"
#include <cstdio>
#include <string>
#include <sys/types.h>



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

void run_sliding_lid_simulation(const double &omega, const double &lidv, const uint &size_x, const uint &size_y, const uint &timesteps, const bool &print) {
  printf("Reynolds Number: %f\n", (lidv * size_x) / ( (1/omega - 0.5)/3));
  Kokkos::Timer timer;
  Kokkos::fence();

  BoltzmanLattice simulation(size_x, size_y, GHOST_BUFFERS, lidv, omega, INIT_SPEED, INIT_SPEED, DENSITY_RHO);
  simulation.open_files("./data/05");

  // Init ghost buffers to 0;
  // Prolly useless, since streaming comes before, but yeah, wont take long anyways
  simulation.init_ghost_buffers(0);
  simulation.feq_distrib();

  // print initial
  if (print)
    simulation.print_velocity(0);

  PRINT_TIMESTEP(0, timesteps);
  for (int i = 1; i <= timesteps; ++i) {
    if (i % 100 == 0) {
      printf("\r");
      PRINT_TIMESTEP(i, timesteps);
    }

    simulation.streaming();
    simulation.bounce_back();
    simulation.collision_fused();

    if (print && i % 20 == 0) 
      simulation.print_velocity(i);
  }
  printf("\n");
  double runtime = timer.seconds();
  PRINT_MLUPS(runtime, size_x, size_y, timesteps);

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
  uint size                 = DEFAULT_SIZE;
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
    if (arg.rfind("--size=", 0) == 0) {
      size = std::stoi(arg.substr(7));
    }
    if (arg.rfind("--N=", 0) == 0) {
      timesteps = std::stoi(arg.substr(4));
    }
    if (arg.rfind("--print", 0) == 0) {
      print = true;
    }
  }
  printf("Omega: %f, Lid Velocity: %f, L: %d, N: %d\n", omega, lid_velocity, size, timesteps);

  Kokkos::initialize(argc, argv);
  run_sliding_lid_simulation(omega, lid_velocity, size, size, timesteps, print);
  Kokkos::finalize();
  return 0;
}
