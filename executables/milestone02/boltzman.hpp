#include <Kokkos_Core.hpp>
#include <fstream>

#define NUM_TIMESTEPS 100
#define NUM_DIRECTIONS 9
#define SIZE_X 15
#define SIZE_Y 15
#define UNIT 1

typedef Kokkos::View<double**>      DENSITY;
typedef Kokkos::View<double***>     DISTRIB;
typedef Kokkos::View<double***>     VELOCITY;

class BoltzmanLattice {
  private:
  std::ofstream dist_file;
  std::ofstream density_file;

  public:
  DISTRIB distribution;
  DISTRIB buffer;
  DENSITY density;
  VELOCITY avg_velocity;
  
  BoltzmanLattice();
  void streaming();
  void init_distribution();
  void calc_density();
  void calc_avg_velocity();
  void open_files();
  void print_dist(uint timestep);
  void print_density(uint timestep);
};
