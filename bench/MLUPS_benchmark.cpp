#include <Eigen/Core>
#include <benchmark/benchmark.h>

#include "boltzman.hpp"

#define MY_BENCHMARK(function)                                                 \
  BENCHMARK(function)->Range(8, 8 << 11)->Complexity(benchmark::oNSquared)

using namespace Eigen;

static void sliding_lid_bench(benchmark::State& state) {
  const int timesteps = 10000;
  const int SIZE = 128;
  BoltzmanLattice simulation(SIZE, SIZE, 1, 0.1, 1.5, 0, 0, 1);

  // Init ghost buffers to 0;
  // Prolly useless, since streaming comes before, but yeah, wont take long anyways
  simulation.init_ghost_buffers(0);
  simulation.feq_distrib();

  // print initial
  for (int i = 1; i <= timesteps; ++i) {
    simulation.streaming();
    simulation.bounce_back();
    simulation.calc_density();
    simulation.calc_avg_velocity();
    simulation.collision();
  }

  state.SetComplexityN(SIZE);
}

MY_BENCHMARK(sliding_lid_bench);

// -----------------------------------------------------------------------------
BENCHMARK_MAIN();
