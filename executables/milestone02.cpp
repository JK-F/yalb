#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <cassert>
#include <gtest/gtest.h>


#define NUM_TIMESTEPS 100
#define OMEGA_RELAXATION 0.1

// Print X-axis labels at the bottom
void run_simulation() {
  BoltzmanLattice simulation(OMEGA_RELAXATION, std::tuple(0.3, 0.3), 0.5);
  simulation.random_distrib();
  simulation.open_files("./data/02dist.csv", "/tmp/doesntmatter");

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


TEST(Milestone02, PeriodicValidation) {
  const double val = 14.1293;
  BoltzmanLattice sim(0.5, std::tuple(0.0, 0.0) ,0.0);
  Direction dir = Direction::RIGHT; 
  for (int x = 0; x < SIZE_X; x++) {
    for (int y = 0; y < SIZE_Y; y++) {
        sim.distribution(0, y, dir) = 0;
    }
  }
  for (int y = 0; y < SIZE_Y; y++) {
      sim.distribution(0, y, dir) = val;
  }
  for (int i = 0; i < SIZE_X; i++) {
    sim.streaming();
  }
  for (int x = 0; x < SIZE_X; x++) {
    for (int y = 0; y < SIZE_Y; y++) {
      if (x == 0)
        assert(sim.distribution(x, y, dir) == val);
      else
        assert(sim.distribution(x, y, dir) == 0);
    }
  }
}


TEST(Milestone02, MassConservation) {
  BoltzmanLattice sim(0.5, std::tuple(0.0, 0.0) ,0.0);
  sim.random_distrib();
  double weight = 0.;
  const auto policy = Kokkos::MDRangePolicy({0,0,0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_reduce(
      policy, [&](const int &x, const int &y, const int &dir, double &acc) {
          acc += sim.distribution(x, y, dir);
      },
      weight);

  for (int i = 0; i < SIZE_X; i++) {
    sim.streaming();
    double running_weight = 0.;
    Kokkos::parallel_reduce(
        policy, [&](const int &x, const int &y, const int &dir, double &acc) {
            acc += sim.distribution(x, y, dir);
        },
        running_weight);
    assert(weight == running_weight);
  }
}
