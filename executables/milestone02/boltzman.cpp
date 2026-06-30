#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "boltzman.hpp"
#include "direction.hpp"

BoltzmanLattice::BoltzmanLattice() {
  std::ofstream dist_file = {};
  std::ofstream density_file = {};

  distribution = DISTRIB { "DISTRIBUTION", SIZE_X, SIZE_Y, NUM_DIRECTIONS };
  buffer = DISTRIB { "BUFFER_DISTRIBUTION", SIZE_X, SIZE_Y, NUM_DIRECTIONS };
  density = DENSITY { "DENSITY", SIZE_X, SIZE_Y };
  avg_velocity = VELOCITY { "VELOCITY", SIZE_X, SIZE_Y, 2};
}

void BoltzmanLattice::streaming() {
  auto policy = Kokkos::MDRangePolicy({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});

  Kokkos::parallel_for("Streaming", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
    auto [new_x, new_y] = updated_coords(x, y, dir);
    buffer(new_x,new_y,dir) = distribution(x, y, dir);
  });
  Kokkos::kokkos_swap(distribution, buffer);
}

void BoltzmanLattice::init_distribution() {
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
  auto generator = random_pool.get_state();
    double val = generator.drand(0., 1000.);
    distribution(x,y,dir) = val;
    random_pool.free_state(generator);
  });
}

void BoltzmanLattice::calc_density() {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("CALC_DENSITY_FOR", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
      double res = 0;
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
         res += distribution(x, y, dir);
      }
      density(x,y) = res;
    });
}

void BoltzmanLattice::calc_avg_velocity() {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("CALC_DENSITY_FOR", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    double rho = density(x, y);
    for (auto dir = 0; dir < NUM_DIRECTIONS; dir++) {
      avg_velocity(x, y, 0) = x_part((Direction) dir) * distribution(x, y, dir) / rho;
      avg_velocity(x, y, 1) = y_part((Direction) dir) * distribution(x, y, dir) / rho;
    }
  });
}

void BoltzmanLattice::open_files() {
  dist_file.open("/home/myo/uni/SoSe26/HighPerformance/yalb/data/dist.csv", std::ios::out);
        dist_file
          << "timestep,"
          << "x,"
          << "y,"
          << "dir,"
          << "dist_value" << std::endl;
  density_file.open("/home/myo/uni/SoSe26/HighPerformance/yalb/data/density.csv", std::ios::out);
        density_file
          << "timestep,"
          << "x,"
          << "y,"
          << "density_value" << std::endl;
}


void BoltzmanLattice::print_dist(uint timestep) {
  for (int x = 0; x < SIZE_X; x++) {
    for (int y = 0; y < SIZE_Y; y++) {
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        dist_file
          << timestep << ","
          << x << ","
          << y << ","
          << dir << ","
          << distribution(x, y, dir) << std::endl;
      }
    }
  }
}

void BoltzmanLattice::print_density(uint timestep) {
  for (int x = 0; x < SIZE_X; x++) {
    for (int y = 0; x < SIZE_Y; y++) {
        density_file 
          << timestep << ","
          << x << ","
          << y << ","
          << density(x, y) << std::endl;
    }
  }
}
