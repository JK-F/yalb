#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "Kokkos_Macros.hpp"
#include "boltzman.hpp"
#include <cassert>

#define EPSILON 0.00001

#define SIZE_X 15
#define SIZE_Y 15

TEST(Milestone02, PeriodicValidation) {
  const double val = 14.1293;
  BoltzmanLattice sim(SIZE_X, SIZE_Y, 0.5, 0.0, 0.0 ,0.0);
  Direction dir = Direction::RIGHT; 
  for (int x = 0; x < sim.size_x; x++) {
    for (int y = 0; y < sim.size_y; y++) {
        sim.distribution(0, y, dir) = 0;
    }
  }
  for (int y = 0; y < sim.size_y; y++) {
      sim.distribution(0, y, dir) = val;
  }
  for (int i = 0; i < sim.size_x; i++) {
    sim.streaming();
  }
  for (int x = 0; x < sim.size_x; x++) {
    for (int y = 0; y < sim.size_y; y++) {
      if (x == 0)
        assert(sim.distribution(x, y, dir) == val);
      else
        assert(sim.distribution(x, y, dir) == 0);
    }
  }
}


TEST(Milestone02, MassConservation) {
  BoltzmanLattice sim(SIZE_X, SIZE_Y, 0.5, 0.0, 0.0 ,0.0);
  auto dist = sim.distribution;
  sim.randomize_distrib();
  double weight = 0.;
  Kokkos::parallel_reduce(
      sim.all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y, double &acc) {
        for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
            acc += dist(x, y, dir);
        }
      },
      weight);

  for (int i = 0; i < sim.size_x; i++) {
    sim.streaming();
    double running_weight = 0.;
    Kokkos::parallel_reduce(
        sim.all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y, double &acc) {
          for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
              acc += dist(x, y, dir);
          }
        },
        running_weight);
    ASSERT_LT(abs(weight - running_weight), EPSILON);
  }
}
