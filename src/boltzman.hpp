#include <Kokkos_Core.hpp>
#include <fstream>
#include <string>
#include <sys/types.h>
#include "direction.hpp"
#include "impl/Kokkos_Profiling.hpp"

typedef Kokkos::View<double**>      DENSITY;
typedef Kokkos::View<double***>     DISTRIB;
typedef Kokkos::View<double**[2]>     VELOCITY;

#define X_DIR 0
#define Y_DIR 1

class BoltzmanLattice {
  private:
  std::string file_prefix;
  std::ofstream dist_file;
  std::ofstream density_file;
  std::ofstream velocity_file;

  public:
  
  DISTRIB distribution;
  DISTRIB buffer;
  DENSITY density;
  VELOCITY avg_velocity;

  uint size_x;
  uint size_y;
  uint ghost_buffers;
  double wall_velocity;
  double omega;
  double rho;
  
  BoltzmanLattice(const uint _size_x, const uint _size_y, const uint _ghost_buffers, const double _wall_velocity, const double _omega, const double ux, const double uy, const double _rho);
  BoltzmanLattice(const uint _size_x, const uint _size_y, const double _omega, const double ux, const double uy, const double _rho);

  void initialize_fields(const double &ux, const double &uy, const double &_rho);
  void shear_wave_init(const double &_eps, const double &_rho);

  void streaming();
  void collision();
  void collision_fused();
  void bounce_back();

  void init_ghost_buffers(double);

  void randomize_distrib();
  void uniform_distrib(double);
  void feq_distrib();

  void calc_density();
  void calc_avg_velocity();


  void open_files(std::string prefix);

  void print_dist(uint timestep);
  void print_density(uint timestep);
  void print_velocity_slice(uint timestep);
  void print_velocity(uint timestep);

  inline Kokkos::MDRangePolicy<Kokkos::Rank<2>> all_nodes_policy() {
    return Kokkos::MDRangePolicy({0 + ghost_buffers, 0 + ghost_buffers}, {size_x - ghost_buffers, size_y - ghost_buffers});
  }

};

KOKKOS_FORCEINLINE_FUNCTION double calc_feq(const uint x, const uint y, const Direction dir, const VELOCITY &velocity, const DENSITY &density) {
    double rho = density(x, y);
    double ux = velocity(x, y, X_DIR);
    double uy = velocity(x, y, Y_DIR);
    double magnitude_squared = ux * ux + uy * uy;

    double w_i = weight(dir);
    double cu = x_part(dir) * ux + y_part(dir) * uy;
    double sum = 1. + 3. * cu + 9. / 2. * cu * cu - 3. / 2. * magnitude_squared;
    return w_i * rho * sum;
}

