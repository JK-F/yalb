#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <sys/types.h>

#define SIZE_X 30
#define SIZE_Y 30

#define NUM_TIMESTEPS 2000
#define OMEGA_RELAXATION 1.5

void run_shear_wave_simulation() {
  BoltzmanLattice simulation(SIZE_X, SIZE_Y, OMEGA_RELAXATION, 0.3, 0.3, 0.5);
  // initialize shear wave


  // Velocity as a shear wave
  // Assert f = feq

  double eps = 0.01;
  double rho = 1;
  simulation.shear_wave_init(eps, rho);

  simulation.feq_distrib();
  simulation.open_files("./data/04");

  // Print initial
  simulation.calc_density();
  simulation.print_dist(0);
  simulation.print_density(0);
  simulation.print_velocity_slice(0);

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();

    simulation.print_dist(i);
    simulation.print_density(i);
    if (i % 300 == 0) {
      simulation.print_velocity_slice(i);
    }

  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  run_shear_wave_simulation();
  Kokkos::finalize();
  return 0;
}
