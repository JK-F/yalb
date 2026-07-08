#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <tuple>

#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

void run_perturbation_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);
  simulation.uniform_distrib(1.0);
  uint denom = 10;
  auto policy = Kokkos::MDRangePolicy({SIZE_X / 2 - SIZE_X / denom , SIZE_Y / 2 - SIZE_Y / denom}, {SIZE_X / 2 + SIZE_X / denom, SIZE_Y / 2 + SIZE_Y / denom});
  Kokkos::parallel_for("INIT_STEP", policy, [&] (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double val = 1.1;
      simulation.distribution(x,y,dir) = val;
    }
  });

  simulation.open_files("./data/03_pertub");

  // Print initial
  simulation.calc_density();
  simulation.print_dist(0);
  simulation.print_density(0);

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();

    simulation.print_dist(i);
    simulation.print_density(i);

  }
}

void run_randomized_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);
  simulation.randomize_distrib();
  simulation.open_files("./data/03_random");

  // Print initial
  simulation.calc_density();
  simulation.print_dist(0);
  simulation.print_density(0);

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();

    simulation.print_dist(i);
    simulation.print_density(i);

  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  run_perturbation_simulation();
  run_randomized_simulation();
  Kokkos::finalize();
  return 0;
}
