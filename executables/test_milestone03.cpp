#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "boltzman.hpp"
#include <cassert>
#define EPSILON 0.00001

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
    assert(abs(weight - running_weight) < EPSILON);
  }
}

TEST(Milestone03, MomentumConservation) {
}

TEST(Milestone03, Relaxation) {
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
