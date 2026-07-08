#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>
#include "direction.hpp"

typedef Kokkos::View<double**>      DENSITY;
typedef Kokkos::View<double***>     DISTRIB;
typedef Kokkos::View<double**[2]>     VELOCITY;

#define X_DIR 0
#define Y_DIR 1

#define SIZE_X 30
#define SIZE_Y 30

class BoltzmanLattice {
  private:
  std::ofstream dist_file;
  std::ofstream density_file;
  std::ofstream velocity_file;

  public:
  
  DISTRIB distribution;
  DISTRIB buffer;
  DENSITY density;
  VELOCITY avg_velocity;

  double omega;
  
  BoltzmanLattice(const double _omega, const std::tuple<double, double> _u, const double _rho);
  void streaming();
  void collision();

  void randomize_distrib();
  void uniform_distrib(double);
  void calc_density();
  void calc_avg_velocity();
  double calc_feq(const uint x, const uint y, const Direction dir);

  void open_files(std::string prefix);

  void print_dist(uint timestep);
  void print_density(uint timestep);
  void print_velocity(uint timestep);
};
