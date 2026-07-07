#include <Kokkos_Core.hpp>
#include "Kokkos_MathematicalConstants.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "boltzman.hpp"
#include <sys/types.h>
#include <tuple>

#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

void run_shear_wave_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);

  // initialize shear wave
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});

  // Velocity as a shear wave
  // Assert f = feq
//
  double eps = 0.01;
  double rho = 1;
  Kokkos::parallel_for("INIT_STEP", policy, [&] (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double k = 2 * Kokkos::numbers::pi / SIZE_Y;

      simulation.density(x, y) = rho;
      simulation.avg_velocity(x, y, Y_DIR) = 0;
      simulation.avg_velocity(x, y, X_DIR) = eps * Kokkos::sin(k * y);
    }
  });

  Kokkos::parallel_for("INIT_STEP", policy, [&] (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double val = simulation.calc_feq(x, y, dir);
      simulation.distribution(x,y,dir) = val;
    }
  });

  simulation.open_files("./data/04_distrib.csv", "./data/04_density.csv");

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
  run_shear_wave_simulation();
  Kokkos::finalize();
  return 0;
}
