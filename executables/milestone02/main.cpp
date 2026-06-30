#include <Kokkos_Core.hpp>
#include "boltzman.hpp"

// Print X-axis labels at the bottom
void run_simulation() {
  BoltzmanLattice simulation;
  simulation.init_distribution();
  simulation.open_files();

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.calc_density();
    simulation.calc_avg_velocity();
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
