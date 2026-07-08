#include <Kokkos_Core.hpp>
#include "boltzman.hpp"
#include <sys/types.h>


#define NUM_TIMESTEPS 700
#define PRINT_TIMESTEP(i) printf("Bolzman Lattice Timestep %05d / %05d\n", i, NUM_TIMESTEPS)

#define SIZE_X 128
#define SIZE_Y 128
#define GHOST_BUFFERS 1

#define DENSITY_RHO 1
#define INIT_SPEED 0
#define LID_VELOCITY 0.1
#define OMEGA_RELAXATION 1.5

void run_sliding_lid_simulation() {
  printf("Reynolds Number: %f\n", (LID_VELOCITY * SIZE_X) / ( (1/OMEGA_RELAXATION - 0.5)/3));

  BoltzmanLattice simulation(SIZE_X, SIZE_Y, GHOST_BUFFERS, LID_VELOCITY, OMEGA_RELAXATION, INIT_SPEED, INIT_SPEED, DENSITY_RHO);
  simulation.open_files("./data/05");

  // Init ghost buffers to 0;
  // Prolly useless, since streaming comes before, but yeah, wont take long anyways
  simulation.init_ghost_buffers(0);
  simulation.feq_distrib();

  // print initial
  simulation.print_dist(0);

  PRINT_TIMESTEP(0);
  for (int i = 1; i <= NUM_TIMESTEPS; ++i) {
    if (i % 100 == 0) {
      PRINT_TIMESTEP(i);
    }

    simulation.streaming();
    simulation.bounce_back();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();

    if (i % 20 == 0) 
      simulation.print_dist(i);
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  run_sliding_lid_simulation();
  Kokkos::finalize();
  return 0;
}
