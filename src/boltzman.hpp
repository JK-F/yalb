#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>
#include "direction.hpp"

typedef Kokkos::View<double**>      DENSITY;
typedef Kokkos::View<double***>     DISTRIB;
typedef Kokkos::View<double***>     VELOCITY;

#define SIZE_X 15
#define SIZE_Y 15

class BoltzmanLattice {
  private:
  std::ofstream dist_file;
  std::ofstream density_file;

  public:
  
  DISTRIB distribution;
  DISTRIB buffer;
  DENSITY density;
  VELOCITY avg_velocity;

  double omega;
  
  BoltzmanLattice(const double _omega, const std::tuple<double, double> _u, const double _rho);
  void streaming();
  void collision();

  void random_distrib();
  void calc_density();
  void calc_avg_velocity();
  double calc_feq(const uint x, const uint y, const Direction dir);

  void open_files(std::string dist_file_name, std::string density_file_name);
  void print_dist(uint timestep);
  void print_density(uint timestep);
};
