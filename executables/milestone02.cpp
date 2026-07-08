#include <Kokkos_Core.hpp>
#include "boltzman.hpp"

#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

#define SIZE_X 15
#define SIZE_Y 15

void run_simulation() {
  BoltzmanLattice simulation(SIZE_X, SIZE_Y, OMEGA_RELAXATION, 0.3, 0.3, 0.5);
  simulation.randomize_distrib();
  simulation.open_files("./data/02");

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.print_dist(i);
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  run_simulation();
  Kokkos::finalize();
  return 0;
}
