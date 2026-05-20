#include <Kokkos_Core.hpp>
#include <iostream>
#include <iomanip>

#define NUM_TIMESTEPS 100
#define NUM_DIRECTIONS 9
#define SIZE_X 15
#define SIZE_Y 10

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
typedef Kokkos::View<Direction**>   VELOCITY;
typedef Kokkos::View<double***>     DISTRIB;


std::tuple<int, int> updated_coords(const int &x, const int &y, const int &dir) {
  int new_x, new_y;
  switch (dir)
  {
    case LEFT:
    case UP_LEFT:
    case DOWN_LEFT:
      new_x = x - 1;
      break;
    case RIGHT:
    case DOWN_RIGHT:
    case UP_RIGHT:
      new_x = x - 1;
      break;
  }
  switch (dir)
  {
    case UP:
    case UP_LEFT:
    case UP_RIGHT:
      new_y = y + 1;
      break;
    case DOWN:
    case DOWN_LEFT:
    case DOWN_RIGHT:
      new_y = y - 1;
      break;
  }
  return std::tuple(new_x, new_y);
}

// Next step:
// Write f^eq s.t.
// sum over all other feq(...) = density
// sum over all other feq(...) = u
// Same also for temp which is constant?!


void init_distribution(DISTRIB f) {
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
    f(x,y,dir) = static_cast<double>(x * y);
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

double calc_velocity(DENSITY density, DISTRIB f, int x, int y) {
  double rho = density(x, y);
  double result;
  Kokkos::parallel_reduce("CALC_DENSITY_REDUCE", NUM_DIRECTIONS, KOKKOS_LAMBDA (const int &dir, double &lsum) {
    lsum += f(x, y, dir);
  }, result);
  return result / rho;
}

void streaming(DISTRIB f) {
  DISTRIB result = DISTRIB("DensityProbabilityFunction", SIZE_X, SIZE_Y, NUM_DIRECTIONS);
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {SIZE_X, SIZE_Y, NUM_DIRECTIONS});
  Kokkos::parallel_for("INIT_DENSITY", policy, KOKKOS_LAMBDA (const int &x, const int &y, const int &dir) {
    auto [new_x, new_y] = updated_coords(x, y, dir);
    if (0 <= new_x && new_x < SIZE_X && 0 <= new_y && new_y < SIZE_Y) {
      result(new_x,new_y,dir) = f(x, y, dir);
    }
  });
  Kokkos::kokkos_swap(f, result);
}

void print_distribution(DISTRIB f, int timestep) {
  // 1. Mirror and bring data to the host (CPU)
  auto f_host = Kokkos::create_mirror_view(f);
  Kokkos::deep_copy(f_host, f);

  // 2. Clear terminal screen and reset cursor (ANSI escape codes)
  // This creates a smooth "live update" animation in your terminal
  std::cout << "\033[2J\033[H"; 
  
  std::cout << "====================================================\n";
  std::cout << " LATTICE BOLTZMANN SIMULATION | TIMESTEP: " << std::setw(3) << timestep << "\n";
  std::cout << "====================================================\n\n";

  // Print top border of the grid
  std::cout << "   +";
  for (int x = 0; x < SIZE_X; ++x) std::cout << "---+";
  std::cout << "\n";

  // 3. Loop through the grid (Y-axis inverted visually so UP is actually up)
  for (int y = SIZE_Y - 1; y >= 0; --y) {
    std::cout << std::setw(2) << y << " |"; // Row index
    
    for (int x = 0; x < SIZE_X; ++x) {
      // Calculate local density (sum of all 9 directions) for this cell
      double local_density = 0.0;
      for (int dir = 0; dir < NUM_DIRECTIONS; ++dir) {
        local_density += f_host(x, y, dir);
      }

      // Choose a character based on the density intensity
      // Adjust thresholds depending on how your feq behaves later!
      char cell_char = ' ';
      if (local_density > 100.0)      cell_char = '#'; // High density
      else if (local_density > 50.0)  cell_char = 'X';
      else if (local_density > 10.0)  cell_char = 'o';
      else if (local_density > 0.0)   cell_char = '.'; // Low density
      else                            cell_char = ' '; // Empty

      // Center the character inside the grid cell block
      std::cout << " " << cell_char << " |";
    }
    
    // Print row separators
    std::cout << "\n   +";
    for (int x = 0; x < SIZE_X; ++x) std::cout << "---+";
    std::cout << "\n";
  }

  // Print X-axis labels at the bottom
  std::cout << "    ";
  for (int x = 0; x < SIZE_X; ++x) {
    std::cout << " " << std::setw(2) << x << " ";
  }
  std::cout << "\n\n";
  
  // 4. Sleep for a moment so human eyes can see the frame update
  // Requires #include <chrono> and #include <thread>
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
}

void run_simulation() {
  DISTRIB f("f", SIZE_X, SIZE_Y, NUM_DIRECTIONS);
  // DENSITY density("density", SIZE_X, SIZE_Y);
  // VELOCITY velocity("velocity", SIZE_X, SIZE_Y);
  init_distribution(f);
  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    print_distribution(f, i);
    streaming(f);
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
