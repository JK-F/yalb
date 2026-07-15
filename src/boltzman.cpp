#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <cassert>
#include <string>
#include "boltzman.hpp"
#include "impl/Kokkos_Profiling.hpp"

BoltzmanLattice::BoltzmanLattice(uint _size_x, uint _size_y, uint _ghost_buffers, double _wall_velocity, double _omega, double ux, double uy, double _rho) {
  // Add ghost buffer at both ends
  size_x = _size_x + 2 * _ghost_buffers;
  size_y = _size_y + 2 * _ghost_buffers;
  ghost_buffers = _ghost_buffers;

  wall_velocity = _wall_velocity;
  omega = _omega;
  rho = _rho;

  distribution = DISTRIB { "DISTRIBUTION", size_x, size_y, NUM_DIRECTIONS };
  buffer = DISTRIB { "BUFFER_DISTRIBUTION", size_x, size_y, NUM_DIRECTIONS };
  density = DENSITY { "DENSITY", size_x, size_y };
  avg_velocity = VELOCITY { "VELOCITY", size_x, size_y};

  this->initialize_fields(ux, uy, _rho);
}

BoltzmanLattice::BoltzmanLattice(uint _size_x, uint _size_y, double _omega, double ux, double uy, double _rho)
  : BoltzmanLattice(_size_x, _size_y, 0, 0, _omega, ux, uy, _rho) {}


void BoltzmanLattice::initialize_fields(const double &ux, const double &uy, const double &_rho) {
  Kokkos::parallel_for(all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    // |u| = sqrt(ux^2 + uy^2) = sqrt(0.09 + 0.09 = 0.18) < 0.5
    avg_velocity(x, y, X_DIR) = ux;
    avg_velocity(x, y, Y_DIR) = uy;
    density(x, y) = _rho;
  });
}

void BoltzmanLattice::shear_wave_init(const double &eps, const double &rho) {
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double k = 2 * Kokkos::numbers::pi / size_y;

      density(x, y) = rho;
      avg_velocity(x, y, X_DIR) = eps * Kokkos::sin(k * y);
      avg_velocity(x, y, Y_DIR) = 0;
    }
  });
}

void BoltzmanLattice::randomize_distrib() {
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      auto generator = random_pool.get_state();
      double val = generator.drand(0., 1000.);
      distribution(x,y,dir) = val;
      random_pool.free_state(generator);
    }
  });
}

void BoltzmanLattice::init_ghost_buffers(double val) {
  assert(ghost_buffers == 1);
  auto policy_x = Kokkos::RangePolicy(0, size_x);
  auto policy_y = Kokkos::RangePolicy(0, size_y);

  Kokkos::parallel_for("init_ghost_x", policy_x, KOKKOS_LAMBDA (const int &x) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distribution(x,0,dir) = val;
      distribution(x,size_y - 1,dir) = val;
      buffer(x,0,dir) = val;
      buffer(x,size_y - 1,dir) = val;
    }
  });
  Kokkos::parallel_for("init_ghost_y", policy_y, KOKKOS_LAMBDA (const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distribution(0,y,dir) = val;
      distribution(size_x -1, y,dir) = val;
      buffer(0,y,dir) = val;
      buffer(size_x -1, y,dir) = val;
    }
  });
}

void BoltzmanLattice::uniform_distrib(double d) {
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint dir = 0; dir < NUM_DIRECTIONS; dir++) {
      distribution(x,y,dir) = d;
    }
  });
}

void BoltzmanLattice::feq_distrib() {
  Kokkos::parallel_for("INIT_STEP", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint d = 0; d < NUM_DIRECTIONS; d++) {
      Direction dir = static_cast<Direction>(d);
      double val = calc_feq(x, y, dir);
      distribution(x,y,dir) = val;
    }
  });
}

void BoltzmanLattice::calc_density() {
  Kokkos::parallel_for("CALC_DENSITY", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
      double res = 0;
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
         res += distribution(x, y, dir);
      }
      density(x,y) = res;
    });
}

void BoltzmanLattice::calc_avg_velocity() {
  Kokkos::parallel_for("CALC_AVG_VELOCITY", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    double rho = density(x, y);
      avg_velocity(x, y, X_DIR) = 0;
      avg_velocity(x, y, Y_DIR) = 0;
    for (auto dir = 0; dir < NUM_DIRECTIONS; dir++) {
      avg_velocity(x, y, X_DIR) += x_part(static_cast<Direction>(dir)) * distribution(x, y, dir) / rho;
      avg_velocity(x, y, Y_DIR) += y_part(static_cast<Direction>(dir)) * distribution(x, y, dir) / rho;
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
  for (int x = 0 + ghost_buffers; x < size_x -ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
      for (int dir = 0; dir < NUM_DIRECTIONS; dir++) {
        dist_file
          << timestep << ","
          << x << ","
          << y << ","
          << dir << ","
          << distribution(x, y, dir) << '\n';
      }
    }
  }
  dist_file.flush();
}

void BoltzmanLattice::print_density(uint timestep) {
  if (!density_file.is_open()) {
    density_file.open(this->file_prefix + "_density.csv", std::ios::out);
    density_file
      << "timestep,"
      << "x,"
      << "y,"
      << "density_value" << std::endl;
  }
  for (int x = 0 + ghost_buffers; x < size_x -ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
        density_file 
          << timestep << ","
          << x << ","
          << y << ","
          << density(x, y) << '\n';
    }
  }
  density_file.flush();
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
  for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
      velocity_file
        << timestep << ","
        << y << ","
        << avg_velocity(size_x/2, y, X_DIR) << '\n';
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
  for (int x = 0 + ghost_buffers; x < size_x - ghost_buffers; x++) {
    for (int y = 0 + ghost_buffers; y < size_y - ghost_buffers; y++) {
        velocity_file
          << timestep << ","
          << x << ","
          << y << ","
          << avg_velocity(x, y, X_DIR)  << ","
          << avg_velocity(x, y, Y_DIR) << '\n';
    }
  }
  velocity_file.flush();
}


void BoltzmanLattice::collision() {
  Kokkos::parallel_for("Collision", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (uint i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      double feq = calc_feq(x, y, static_cast<Direction>(dir));
      distribution(x, y, dir) += omega * (feq - distribution(x, y, dir)) ;
    }
  });
}

void BoltzmanLattice::streaming() {
  Kokkos::parallel_for("Streaming", all_nodes_policy(), KOKKOS_LAMBDA (const int &x, const int &y) {
    for (int i = 0; i < NUM_DIRECTIONS; i++) {
      auto dir = static_cast<Direction>(i);
      auto [new_x, new_y] = updated_coords(x, y, dir, size_x, size_y);
      buffer(new_x,new_y,dir) = distribution(x, y, dir);
    }
  });
  Kokkos::kokkos_swap(distribution, buffer);
}

void BoltzmanLattice::bounce_back() {
  assert(ghost_buffers == 1);
  auto policy_x = Kokkos::RangePolicy(1, size_x - 1);
  auto policy_y = Kokkos::RangePolicy(1, size_y - 1);

  Kokkos::parallel_for("bounce_back_x", policy_x, KOKKOS_LAMBDA (const int &x) {
      // Lower border
      distribution(x, 1, UP)                    = distribution(x,     0, DOWN);
      distribution(x, 1, UP_LEFT)               = distribution(x + 1, 0, DOWN_RIGHT);
      distribution(x, 1, UP_RIGHT)              = distribution(x - 1, 0, DOWN_LEFT);
      
      // Upper border (sliding lid)
      // Assume contant density 
      // Use (c_s)^2 = 1/3 => 1/(c_s)^2 = 3, and 2 * 3 = 6
      const double partial_prod = 6 * rho;

      // We only have x velocity on the wall
      distribution(x, size_y - 2, DOWN)         = distribution(x    , size_y - 1, UP)       - partial_prod * weight(UP)         * x_part(UP)        * wall_velocity;
      distribution(x, size_y - 2, DOWN_LEFT)    = distribution(x + 1, size_y - 1, UP_RIGHT) - partial_prod * weight(UP_RIGHT)   * x_part(UP_RIGHT)  * wall_velocity;
      distribution(x, size_y - 2, DOWN_RIGHT)   = distribution(x - 1, size_y - 1, UP_LEFT)  - partial_prod * weight(UP_LEFT)    * x_part(UP_LEFT)   * wall_velocity;
      
  });

  // Im chosing to have the sliding lid overwritten by simple bounce back
  // (See cases for DOWN_LEFT and DOWN_RIGHT)
  Kokkos::parallel_for("bounce_back_y", policy_y, KOKKOS_LAMBDA (const int &y) {
      // Left border
      distribution(1,           y, RIGHT)        = distribution(0,           y,          LEFT);
      distribution(1,           y, UP_RIGHT)     = distribution(0,           y - 1,      DOWN_LEFT);
      distribution(1,           y, DOWN_RIGHT)   = distribution(0,           y + 1,      UP_LEFT);
      
      // Right border
      distribution(size_x - 2,  y, LEFT)         = distribution(size_x - 1,  y,          RIGHT);
      distribution(size_x - 2,  y, UP_LEFT)      = distribution(size_x - 1,  y - 1,      DOWN_RIGHT);
      distribution(size_x - 2,  y, DOWN_LEFT)    = distribution(size_x - 1,  y + 1,      UP_RIGHT);
  });
}
