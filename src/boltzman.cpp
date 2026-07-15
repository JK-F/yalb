#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cassert>
#include <iostream>
#include <string>
#include "boltzman.hpp"
#include "impl/Kokkos_Profiling.hpp"

BoltzmanLattice::BoltzmanLattice(uint _size_x, uint _size_y, uint _ghost_buffers, double _wall_velocity, double _omega, double ux, double uy, double _rho) {
  // Add ghost this->buffer at both ends
  size_x = _size_x + 2 * _ghost_buffers;
  size_y = _size_y + 2 * _ghost_buffers;
  ghost_buffers = _ghost_buffers;

  wall_velocity = _wall_velocity;
  omega = _omega;
  rho = _rho;

  this->distribution = DISTRIB { "DISTRIBUTION", size_x, size_y, NUM_DIRECTIONS };
  this->buffer = DISTRIB { "BUFFER_DISTRIBUTION", size_x, size_y, NUM_DIRECTIONS };
  this->density = DENSITY { "DENSITY", size_x, size_y };
  this->avg_velocity = VELOCITY { "VELOCITY", size_x, size_y};

  this->initialize_fields(ux, uy, _rho);
}

BoltzmanLattice::BoltzmanLattice(uint _size_x, uint _size_y, double _omega, double ux, double uy, double _rho)
  : BoltzmanLattice(_size_x, _size_y, 0, 0, _omega, ux, uy, _rho) {}


void BoltzmanLattice::initialize_fields(const double &ux, const double &uy, const double &_rho) {
  auto dens = this->density;
  auto velocity = this->avg_velocity;
  Kokkos::parallel_for(all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    // |u| = sqrt(ux^2 + uy^2) = sqrt(0.09 + 0.09 = 0.18) < 0.5
    velocity(x, y, X_DIR) = ux;
    velocity(x, y, Y_DIR) = uy;
    dens(x, y) = _rho;
  });
}

void BoltzmanLattice::shear_wave_init(const double &eps, const double &rho) {
  auto dens = this->density;
  auto velocity = this->avg_velocity;
  auto sy = size_y;
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      double k = 2 * Kokkos::numbers::pi / sy;

      dens(x, y) = rho;
      velocity(x, y, X_DIR) = eps * Kokkos::sin(k * y);
      velocity(x, y, Y_DIR) = 0;
    }
  });
}

void BoltzmanLattice::randomize_distrib() {
  auto distrib = this->distribution;
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      auto generator = random_pool.get_state();
      double val = generator.drand(0., 1000.);
      distrib(x,y,dir) = val;
      random_pool.free_state(generator);
    }
  });
}

void BoltzmanLattice::init_ghost_buffers(double val) {
  assert(ghost_buffers == 1);
  auto policy_x = Kokkos::RangePolicy(0, size_x);
  auto policy_y = Kokkos::RangePolicy(0, size_y);
  auto distrib = this->distribution;
  auto buff = this->buffer;
  auto sx = size_x;
  auto sy = size_y;

  Kokkos::parallel_for("init_ghost_x", policy_x, KOKKOS_LAMBDA (const int &x) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distrib(x,0,dir) = val;
      distrib(x,sy - 1,dir) = val;
      buff(x,0,dir) = val;
      buff(x,sy - 1,dir) = val;
    }
  });
  Kokkos::parallel_for("init_ghost_y", policy_y, KOKKOS_LAMBDA (const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distrib(0,y,dir) = val;
      distrib(sx -1, y,dir) = val;
      buff(0,y,dir) = val;
      buff(sx -1, y,dir) = val;
    }
  });
}

void BoltzmanLattice::uniform_distrib(double d) {
  auto distrib = this->distribution;
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distrib(x,y,dir) = d;
    }
  });
}

void BoltzmanLattice::feq_distrib() {
  auto distrib = this->distribution;
  auto avg_velocity = this->avg_velocity;
  auto density = this->density;
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double val = calc_feq(x, y, dir, avg_velocity, density);
      distrib(x,y,dir) = val;
    }
  });
}

void BoltzmanLattice::calc_density() {
  auto dens = this->density;
  auto distrib = this->distribution;
  Kokkos::parallel_for("CALC_DENSITY", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
      double res = 0;
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
         res += distrib(x, y, dir);
      }
      dens(x,y) = res;
    });
}

void BoltzmanLattice::calc_avg_velocity() {
  auto velocity = this->avg_velocity;
  auto distrib = this->distribution;
  auto dens = this->density;
  Kokkos::parallel_for("CALC_AVG_VELOCITY", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    double rho = dens(x, y);
      velocity(x, y, X_DIR) = 0;
      velocity(x, y, Y_DIR) = 0;
    for (auto dir = 0; dir < NUM_DIRECTIONS; dir++) {
      velocity(x, y, X_DIR) += x_part(static_cast<Direction>(dir)) * distrib(x, y, dir) / rho;
      velocity(x, y, Y_DIR) += y_part(static_cast<Direction>(dir)) * distrib(x, y, dir) / rho;
    }
  });
}

void BoltzmanLattice::open_files(std::string prefix) {
  this->file_prefix = prefix;
}

void BoltzmanLattice::print_dist(uint timestep) {
  if (!dist_file.is_open()) {
    dist_file.open(this->file_prefix + "_distrib.csv", std::ios::out);
    dist_file
      << "timestep,"
      << "x,"
      << "y,"
      << "dir,"
      << "dist_value" << '\n';
  }

  auto mirror = Kokkos::create_mirror_view(this->distribution);
  Kokkos::deep_copy(mirror, this->distribution);
  for (int x = 0 + ghost_buffers; x < size_x -ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        dist_file
          << timestep << ","
          << x << ","
          << y << ","
          << dir << ","
          << mirror(x, y, dir) << '\n';
      }
    }
  }
  dist_file.flush();
}

void BoltzmanLattice::print_density(uint timestep) {
  if (!this->density_file.is_open()) {
    this->density_file.open(this->file_prefix + "_density.csv", std::ios::out);
    this->density_file
      << "timestep,"
      << "x,"
      << "y,"
      << "this->density_value" << std::endl;
  }
  auto mirror = Kokkos::create_mirror_view(this->density);
  Kokkos::deep_copy(mirror, this->density);
  for (int x = 0 + ghost_buffers; x < size_x -ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
        this->density_file 
          << timestep << ","
          << x << ","
          << y << ","
          << mirror(x, y) << '\n';
    }
  }
  this->density_file.flush();
}

void BoltzmanLattice::print_velocity_slice(uint timestep) {
  if (!velocity_file.is_open()) {
    velocity_file.open(this->file_prefix + "_velocity_slice.csv", std::ios::out);
    velocity_file
      << "timestep,"
      << "y,"
      << "velocity_value" << std::endl;
  }
  // Print all ux for all y and a fix x = N/2
  auto mirror = Kokkos::create_mirror_view(this->avg_velocity);
  Kokkos::deep_copy(mirror, this->avg_velocity);
  for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
      velocity_file
        << timestep << ","
        << y << ","
        << mirror(size_x/2, y, X_DIR) << '\n';
  }
  velocity_file.flush();
}

void BoltzmanLattice::print_velocity(uint timestep) {
  if (!velocity_file.is_open()) {
    velocity_file.open(this->file_prefix + "_velocity.csv", std::ios::out);
    velocity_file
      << "timestep,"
      << "x,"
      << "y,"
      << "ux,"
      << "uy" << std::endl;
  }
  // Print all ux for all y and a fix x = N/2
  auto mirror = Kokkos::create_mirror_view(this->avg_velocity);
  Kokkos::deep_copy(mirror, this->avg_velocity);
  for (int x = 0 + ghost_buffers; x < size_x - ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
        velocity_file
          << timestep << ","
          << x << ","
          << y << ","
          << mirror(x, y, X_DIR)  << ","
          << mirror(x, y, Y_DIR) << '\n';
    }
  }
  velocity_file.flush();
}


void BoltzmanLattice::collision() {
  auto distrib = this->distribution;
  auto avg_velocity = this->avg_velocity;
  auto density = this->density;
  auto om = omega;
  Kokkos::parallel_for("Collision", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      double feq = calc_feq(x, y, static_cast<Direction>(dir), avg_velocity, density);
      distrib(x, y, dir) += om * (feq - distrib(x, y, dir)) ;
    }
  });
}

void BoltzmanLattice::collision_fused() {
  auto distrib = this->distribution;
  auto density = this->density;
  auto velocity = this->avg_velocity;
  auto om = omega;
  Kokkos::parallel_for("CollisionFused", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    double f[NUM_DIRECTIONS];
    double rho = 0;
    double ux = 0;
    double uy = 0;
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
      double fi = distrib(x, y, i);
      f[i] = fi;
      rho += fi;
      Direction d = static_cast<Direction>(i);
      ux += x_part(d) * fi;
      uy += y_part(d) * fi;
    }
    ux /= rho;
    uy /= rho;
    density(x, y) = rho;
    velocity(x, y, X_DIR) = ux;
    velocity(x, y, Y_DIR) = uy;
    double magnitude_squared = ux * ux + uy * uy;
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
      Direction d = static_cast<Direction>(i);
      double w_i = weight(d);
      double cu = x_part(d) * ux + y_part(d) * uy;
      double sum = 1. + 3. * cu + 9. / 2. * cu * cu - 3. / 2. * magnitude_squared;
      double feq = w_i * rho * sum;
      distrib(x, y, i) = f[i] + om * (feq - f[i]);
    }
  });
}

void BoltzmanLattice::streaming() {
  auto buff = this->buffer;
  auto distrib = this->distribution;
  auto sx = size_x;
  auto sy = size_y;
  Kokkos::parallel_for("Streaming", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      auto coords = updated_coords(x, y, dir, sx, sy);
      buff(coords.first, coords.second,dir) = distrib(x, y, dir);
    }
  });
  Kokkos::kokkos_swap(this->distribution, this->buffer);
}

void BoltzmanLattice::bounce_back() {
  assert(ghost_buffers == 1);
  auto policy_x = Kokkos::RangePolicy(1, size_x - 1);
  auto policy_y = Kokkos::RangePolicy(1, size_y - 1);

  auto distrib = this->distribution;
  auto rho_local = rho;
  auto wv = wall_velocity;
  auto sx = size_x;
  auto sy = size_y;

  Kokkos::parallel_for("bounce_back_x", policy_x, KOKKOS_LAMBDA (const int &x) {
      // Lower border
      distrib(x, 1, UP)                    = distrib(x,     0, DOWN);
      distrib(x, 1, UP_LEFT)               = distrib(x + 1, 0, DOWN_RIGHT);
      distrib(x, 1, UP_RIGHT)              = distrib(x - 1, 0, DOWN_LEFT);
      
      // Upper border (sliding lid)
      // Assume contant density 
      // Use (c_s)^2 = 1/3 => 1/(c_s)^2 = 3, and 2 * 3 = 6
      const double partial_prod = 6 * rho_local;

      // We only have x velocity on the wall
      distrib(x, sy - 2, DOWN)         = distrib(x    , sy - 1, UP)       - partial_prod * weight(UP)         * x_part(UP)        * wv;
      distrib(x, sy - 2, DOWN_LEFT)    = distrib(x + 1, sy - 1, UP_RIGHT) - partial_prod * weight(UP_RIGHT)   * x_part(UP_RIGHT)  * wv;
      distrib(x, sy - 2, DOWN_RIGHT)   = distrib(x - 1, sy - 1, UP_LEFT)  - partial_prod * weight(UP_LEFT)    * x_part(UP_LEFT)   * wv;
      
  });

  // Im chosing to have the sliding lid overwritten by simple bounce back
  // (See cases for DOWN_LEFT and DOWN_RIGHT)
  Kokkos::parallel_for("bounce_back_y", policy_y, KOKKOS_LAMBDA (const int &y) {
      // Left border
      distrib(1,           y, RIGHT)        = distrib(0,           y,          LEFT);
      distrib(1,           y, UP_RIGHT)     = distrib(0,           y - 1,      DOWN_LEFT);
      distrib(1,           y, DOWN_RIGHT)   = distrib(0,           y + 1,      UP_LEFT);
      
      // Right border
      distrib(sx - 2,  y, LEFT)         = distrib(sx - 1,  y,          RIGHT);
      distrib(sx - 2,  y, UP_LEFT)      = distrib(sx - 1,  y - 1,      DOWN_RIGHT);
      distrib(sx - 2,  y, DOWN_LEFT)    = distrib(sx - 1,  y + 1,      UP_RIGHT);
  });
}
