#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <string>
#include <sys/types.h>


#define DEFAULT_TIMESTEPS 2000
#define PRINT_TIMESTEP(i, timesteps) printf("Bolzman Lattice Timestep %05d / %05d\n", i, timesteps)

#define DEFAULT_SIZE 30
#define SIZE_X DEFAULT_SIZE
#define SIZE_Y DEFAULT_SIZE
#define GHOST_BUFFERS 1

#define DENSITY_RHO 1
#define INIT_SPEED 0
#define DEFAULT_LIDV 0.1
#define DEFAULT_OMEGA 1.5

void run_sliding_lid_simulation(const double &omega, const double &lidv, const uint &size_x, const uint &size_y, const uint &timesteps) {
  printf("Reynolds Number: %f\n", (lidv * size_x) / ( (1/omega - 0.5)/3));

  BoltzmanLattice simulation(size_x, size_y, GHOST_BUFFERS, lidv, omega, INIT_SPEED, INIT_SPEED, DENSITY_RHO);
  simulation.open_files("./data/05");

  // Init ghost buffers to 0;
  // Prolly useless, since streaming comes before, but yeah, wont take long anyways
  simulation.init_ghost_buffers(0);
  simulation.feq_distrib();

  // print initial
  simulation.print_velocity(0);

  PRINT_TIMESTEP(0, timesteps);
  for (int i = 1; i <= timesteps; ++i) {
    if (i % 100 == 0) {
      PRINT_TIMESTEP(i, timesteps);
    }

    simulation.streaming();
    simulation.bounce_back();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();

    if (i % 20 == 0) 
      simulation.print_velocity(i);
  }
}

int main(int argc, char* argv[]) {
  double omega              = DEFAULT_OMEGA;
  double lid_velocity       = DEFAULT_LIDV;
  uint size                 = DEFAULT_SIZE;
  uint timesteps            = DEFAULT_TIMESTEPS;
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
      timesteps = std::stoi(arg.substr(5));
    }
  }
  printf("Omega: %f, Lid Velocity: %f, L: %d\n", omega, lid_velocity, size);
  return 0;
  Kokkos::initialize(argc, argv);
  run_sliding_lid_simulation(omega, lid_velocity, size, size, timesteps);
  Kokkos::finalize();
  return 0;
}
