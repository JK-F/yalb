#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>
#include "boltzman.hpp"
#include "impl/Kokkos_Profiling.hpp"
#include <cassert>
#define EPSILON 0.00001

TEST(Milestone03, MassConservation) {
  BoltzmanLattice sim(0.5, std::tuple(0.3, 0.3), 0.1);
  sim.randomize_distrib();
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
    ASSERT_LT(abs(weight - running_weight), EPSILON);
  }
}

TEST(Milestone03, MomentumConservation) {
  BoltzmanLattice sim(0.5, std::tuple(0.3, 0.3), 0.1);
  sim.randomize_distrib();
  const auto policy_2d = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});
  const auto policy_3d = Kokkos::MDRangePolicy({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});

  for (int i = 1; i <= 100; ++i) {
    sim.streaming();
    sim.calc_density();
    sim.calc_avg_velocity();

    double rho_u_x = 0, rho_u_y = 0;
    Kokkos::parallel_reduce(
        policy_2d, [&](int x, int y, double& mx, double& my) {
            mx += sim.density(x, y) * sim.avg_velocity(x, y, 0);
            my += sim.density(x, y) * sim.avg_velocity(x, y, 1);
        }, Kokkos::Sum<double>(rho_u_x), Kokkos::Sum<double>(rho_u_y));

    double c_f_x = 0, c_f_y = 0;
    Kokkos::parallel_reduce(
        policy_3d, [&](int x, int y, int dir, double& mx, double& my) {
            Direction d = static_cast<Direction>(dir);
            double f = sim.distribution(x, y, dir);
            mx += x_part(d) * f;
            my += y_part(d) * f;
        }, Kokkos::Sum<double>(c_f_x), Kokkos::Sum<double>(c_f_y));

    ASSERT_LT(abs(rho_u_x - c_f_x), EPSILON);
    ASSERT_LT(abs(rho_u_y - c_f_y), EPSILON);

    sim.collision();
    sim.calc_density();
    sim.calc_avg_velocity();

    Kokkos::parallel_reduce(
        policy_2d, [&](int x, int y, double& mx, double& my) {
            mx += sim.density(x, y) * sim.avg_velocity(x, y, 0);
            my += sim.density(x, y) * sim.avg_velocity(x, y, 1);
        }, Kokkos::Sum<double>(rho_u_x), Kokkos::Sum<double>(rho_u_y));

    Kokkos::parallel_reduce(
        policy_3d, [&](int x, int y, int dir, double& mx, double& my) {
            Direction d = static_cast<Direction>(dir);
            double f = sim.distribution(x, y, dir);
            mx += x_part(d) * f;
            my += y_part(d) * f;
        }, Kokkos::Sum<double>(c_f_x), Kokkos::Sum<double>(c_f_y));

    ASSERT_LT(abs(rho_u_x - c_f_x), EPSILON);
    ASSERT_LT(abs(rho_u_y - c_f_y), EPSILON);
  }
}

TEST(Milestone03, EquilibriumFixpoint) {
  const double u = 0.3;
  const double rho = 0.1;
  BoltzmanLattice sim(0.5, std::tuple(u, u), rho);
  for (int x = 0; x < SIZE_X; x++) {
    for (int y = 0; y < SIZE_Y; y++) {
      for (int d = 0; d < NUM_DIRECTIONS; d++) {
        auto dir = static_cast<Direction>(d);
        sim.distribution(x, y, d) = sim.calc_feq(x, y, dir);
      }
    }
  }

  for (int i = 1; i <= 100; ++i) {
    sim.streaming();
    sim.calc_density();
    sim.calc_avg_velocity();
    sim.collision();
    for (int x = 0; x < SIZE_X; x++) {
      for (int y = 0; y < SIZE_Y; y++) {
        ASSERT_LT(abs(u - sim.avg_velocity(x, y, 0)), EPSILON);
        ASSERT_LT(abs(u - sim.avg_velocity(x, y, 1)), EPSILON);
        ASSERT_LT(abs(rho - sim.density(x, y)), EPSILON);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  Kokkos::initialize(argc, argv);
  int result = RUN_ALL_TESTS();
  Kokkos::finalize();
  return result;
}
