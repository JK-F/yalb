#include <Kokkos_Core.hpp>
#include "boltzman.hpp"

#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

// Print X-axis labels at the bottom
void run_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);
  simulation.random_distrib();
  simulation.open_files();

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();
    simulation.print_dist(i);
  }
}

int main(int argc, char* argv[]) {
  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  // Run the simul1ation
  run_simulation();

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
