#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "boltzman.hpp"
#include <cassert>

#define EPSILON 0.00001

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
  sim.randomize_distrib();
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
    ASSERT_LT(abs(weight - running_weight), EPSILON);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
