#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <cassert>
#include <gtest/gtest.h>
#include <tuple>

#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

// Print X-axis labels at the bottom
void run_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);
  simulation.random_distrib();
  simulation.open_files("./data/03distrib.csv", "./data/03density.csv");

  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    simulation.streaming();
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

TEST(Milestone03, MassConservation) {
  BoltzmanLattice sim(0.5, std::tuple(0.3, 0.3), 0.1);
  sim.random_distrib();
  double weight = 0.;
  const auto policy = Kokkos::MDRangePolicy({0,0,0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_reduce(
      policy, [&](const int &x, const int &y, const int &dir, double &acc) {
          acc += sim.distribution(x, y, dir);
      },
      weight);

  for (int i = 1; i <= 100; ++i) {
    sim.streaming();
    sim.calc_density();
    sim.calc_avg_velocity();
    sim.collision();
    double running_weight = 0.;
    Kokkos::parallel_reduce(
        policy, [&](const int &x, const int &y, const int &dir, double &acc) {
            acc += sim.distribution(x, y, dir);
        },
        running_weight);
    assert(weight == running_weight);
  }
}

TEST(Milestone03, MomentumConservation) {
}

TEST(Milestone03, Relaxation) {
}
