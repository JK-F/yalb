#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "boltzman.hpp"

BoltzmanLattice::BoltzmanLattice(double _omega, std::tuple<double, double> _u, double _rho) {
  std::ofstream dist_file = {};
  std::ofstream density_file = {};

  distribution = DISTRIB { "DISTRIBUTION", SIZE_X, SIZE_Y, NUM_DIRECTIONS };
  buffer = DISTRIB { "BUFFER_DISTRIBUTION", SIZE_X, SIZE_Y, NUM_DIRECTIONS };
  density = DENSITY { "DENSITY", SIZE_X, SIZE_Y };
  avg_velocity = VELOCITY { "VELOCITY", SIZE_X, SIZE_Y};
  omega = _omega;
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for(policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    // |u| = sqrt(ux^2 + uy^2) = sqrt(0.09 + 0.09 = 0.18) < 0.5
    auto [ux, uy] = _u;
    avg_velocity(x, y, 0) = ux;
    avg_velocity(x, y, 1) = uy;
    density(x, y) = _rho;
  });
}

void BoltzmanLattice::streaming() {
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});

  Kokkos::parallel_for("Streaming", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      auto [new_x, new_y] = updated_coords(x, y, dir, SIZE_X, SIZE_Y);
      buffer(new_x,new_y,dir) = distribution(x, y, dir);
    }
  });
  Kokkos::kokkos_swap(distribution, buffer);
}

void BoltzmanLattice::randomize_distrib() {
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("INIT_STEP", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      auto generator = random_pool.get_state();
      double val = generator.drand(0., 1000.);
      distribution(x,y,dir) = val;
      random_pool.free_state(generator);
    }
  });
}

void BoltzmanLattice::uniform_distrib(double d) {
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("INIT_STEP", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distribution(x,y,dir) = d;
    }
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
      avg_velocity(x, y, 0) = 0;
      avg_velocity(x, y, 1) = 0;
    for (auto dir = 0; dir < NUM_DIRECTIONS; dir++) {
      avg_velocity(x, y, 0) += x_part(static_cast<Direction>(dir)) * distribution(x, y, dir) / rho;
      avg_velocity(x, y, 1) += y_part(static_cast<Direction>(dir)) * distribution(x, y, dir) / rho;
    }
  });
}

void BoltzmanLattice::open_files(std::string dist_file_name, std::string density_file_name) {
  dist_file.open(dist_file_name, std::ios::out);
        dist_file
          << "timestep,"
          << "x,"
          << "y,"
          << "dir,"
          << "dist_value" << std::endl;
  density_file.open(density_file_name, std::ios::out);
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
    for (int y = 0; y < SIZE_Y; y++) {
        density_file 
          << timestep << ","
          << x << ","
          << y << ","
          << density(x, y) << std::endl;
    }
  }
}

inline double BoltzmanLattice::calc_feq(const uint x, const uint y, const Direction dir) {
    double rho = density(x, y);
    double ux = avg_velocity(x, y, 0);
    double uy = avg_velocity(x, y, 1);
    double magnitude_squared = ux * ux + uy * uy;

    double w_i = weight(dir);
    double cu = x_part(dir) * ux + y_part(dir) * uy;
    double sum = 1. + 3. * cu + 9. / 2. * cu * cu - 3. / 2. * magnitude_squared;
    return w_i * rho * sum;
}

void BoltzmanLattice::collision() {
  auto policy = Kokkos::MDRangePolicy({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("Streaming", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      double feq = calc_feq(x, y, static_cast<Direction>(dir));
      distribution(x, y, dir) += omega * (feq - distribution(x, y, dir)) ;
    }
  });
}

