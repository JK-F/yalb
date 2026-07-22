#include <mpi.h>
#include <direction.hpp>

inline int neighbors_rank(int rank, MPI_Comm &cart, Direction dir) {
  int coords[2], neighbor_rank;
  int dims[2], periods[2];

  MPI_Cart_get(cart, 2, dims, periods, coords);
  coords[0] += static_cast<int>(x_part(dir));
  coords[1] += static_cast<int>(y_part(dir));

  if (!periods[0] && (0 > coords[0] || coords[0] >= dims[0])) {
    return MPI_PROC_NULL;
  }
  if (!periods[1] && (0 > coords[1] || coords[1] >= dims[1])) {
    return MPI_PROC_NULL;
  }

  if (MPI_Cart_rank(cart, coords, &neighbor_rank) != MPI_SUCCESS) {
    return MPI_PROC_NULL;
  }

  return neighbor_rank;
}
