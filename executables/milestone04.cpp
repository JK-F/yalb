#include <Kokkos_Core.hpp>
#include "Kokkos_MathematicalFunctions.hpp"
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
  Kokkos::parallel_for("INIT_STEP", simulation.all_nodes_policy(), [&] (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double k = 2 * Kokkos::numbers::pi / SIZE_Y;

      simulation.density(x, y) = rho;
      simulation.avg_velocity(x, y, X_DIR) = eps * Kokkos::sin(k * y);
      simulation.avg_velocity(x, y, Y_DIR) = 0;
    }
  });

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
