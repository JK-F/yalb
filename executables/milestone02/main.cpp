#include "Kokkos_Printf.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#define NUM_TIMESTEPS 100
#define NUM_DIRECTIONS 9
#define SIZE_X 15
#define SIZE_Y 10
#define UNIT 1

enum Direction {
  // No Y movement
  NONE = 0,
  RIGHT = 1,
  LEFT = 3,
  // No X Movement
  UP = 2,
  DOWN = 4,
  // Diagonals
  UP_RIGHT = 5,
  UP_LEFT = 6,
  DOWN_LEFT = 7,
  DOWN_RIGHT = 8,
};

typedef Kokkos::View<double**>      DENSITY;
typedef Kokkos::View<double***>     DISTRIB;
typedef Kokkos::View<double**>      VELOCITY;


std::tuple<int, int> updated_coords(const int &x, const int &y, const int &dir) {
  int new_x, new_y;
  switch (dir)
  {
    case LEFT:
    case UP_LEFT:
    case DOWN_LEFT:
      new_x = x - UNIT;
      break;
    case RIGHT:
    case DOWN_RIGHT:
    case UP_RIGHT:
      new_x = x + UNIT;
      break;
  }
  switch (dir)
  {
    case UP:
    case UP_LEFT:
    case UP_RIGHT:
      new_y = y + UNIT;
      break;
    case DOWN:
    case DOWN_LEFT:
    case DOWN_RIGHT:
      new_y = y - UNIT;
      break;
  }
  return std::tuple((new_x + SIZE_X) % SIZE_X, (new_y + SIZE_Y) % SIZE_Y);
}

// Next step:
// Write f^eq s.t.
// sum over all other feq(...) = density
// sum over all other feq(...) = u
// Same also for temp which is constant?!


void init_distribution(DISTRIB f) {
  Kokkos::Random_XorShift64_Pool<> random_pool(/*seed=*/12345);
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
    auto generator = random_pool.get_state();
    double val = generator.drand(0., 1000.);
    f(x,y,dir) = val;
    random_pool.free_state(generator);
  });
}

// Calc is short for calculator

void calc_density(DENSITY result, DISTRIB f) {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("CALC_DENSITY_FOR", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    Kokkos::parallel_reduce("CALC_DENSITY_REDUCE", NUM_DIRECTIONS, KOKKOS_LAMBDA (const int &dir, double &lsum) {
      lsum += f(x, y, dir);
    }, result(x,y));
  });
}

inline const double y_part(Direction direction) {
  switch (direction) {
  case UP:
  case UP_RIGHT:
  case UP_LEFT:
    return 1;
  case DOWN:
  case DOWN_LEFT:
  case DOWN_RIGHT:
    return -1;
  case NONE:
  case RIGHT:
  case LEFT:
    return 0;
  }
}

inline const double x_part(Direction direction) {
  switch (direction) {
  case RIGHT:
  case DOWN_RIGHT:
  case UP_RIGHT:
    return 1;
  case LEFT:
  case DOWN_LEFT:
  case UP_LEFT:
    return -1;
  case NONE:
  case UP:
  case DOWN:
    return 0;
  }
}

void calc_avg_velocity(VELOCITY vel_x, VELOCITY vel_y, DENSITY density, DISTRIB f) {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {SIZE_X, SIZE_Y});
  Kokkos::parallel_for("CALC_DENSITY_FOR", policy, KOKKOS_LAMBDA (const int &x, const int &y) {
    double rho = density(x, y);
    for (auto dir = 0; dir < NUM_DIRECTIONS; dir++) {
      vel_y(x, y) = y_part((Direction) dir) * f(x, y, dir) / rho;
      vel_x(x, y) = x_part((Direction) dir) * f(x, y, dir) / rho;
    }
  });
}

void streaming(DISTRIB f) {
  DISTRIB result = DISTRIB("DensityProbabilityFunction", SIZE_X, SIZE_Y, NUM_DIRECTIONS);
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});

  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
    auto [new_x, new_y] = updated_coords(x, y, dir);
    result(new_x,new_y,dir) = f(x, y, dir);
  });
  Kokkos::kokkos_swap(f, result);
}

void print_dist(DISTRIB f, uint timestep) {
  Kokkos::printf("\"%d\": [", timestep);
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
      Kokkos::printf("[%d, %d, %d, %f],", x, y, dir, f(x, y, dir));
  });
  Kokkos::printf("],");
}

// Print X-axis labels at the bottom
void run_simulation() {
  DISTRIB f("f", SIZE_X, SIZE_Y, NUM_DIRECTIONS);
  DENSITY density("density", SIZE_X, SIZE_Y);
  VELOCITY vel_x("X Velocity", SIZE_X, SIZE_Y);
  VELOCITY vel_y("Y Velocity", SIZE_X, SIZE_Y);
  calc_density(density, f);
  init_distribution(f);
  Kokkos::printf("{");
  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    streaming(f);
    calc_density(density, f);
    calc_avg_velocity(vel_x, vel_y, density, f);
    print_dist(f, i);
  }
  Kokkos::printf("}");
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
