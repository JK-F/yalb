#include <Kokkos_Core.hpp>
#define NUM_DIRECTIONS 9

enum Direction {
  NONE = 0,
  RIGHT = 1,
  LEFT = 3,
  UP = 2,
  DOWN = 4,
  UP_RIGHT = 5,
  UP_LEFT = 6,
  DOWN_LEFT = 7,
  DOWN_RIGHT = 8,
};

KOKKOS_FORCEINLINE_FUNCTION
double x_part(Direction direction) {
  switch (direction) {
    case NONE: return 0;
    case RIGHT: return 1;
    case UP: return 0;
    case LEFT: return -1;
    case DOWN: return 0;
    case UP_RIGHT: return 1;
    case UP_LEFT: return -1;
    case DOWN_LEFT: return -1;
    case DOWN_RIGHT: return 1;
  }
  return 0;
}

KOKKOS_FORCEINLINE_FUNCTION
double y_part(Direction direction) {
  switch (direction) {
    case NONE: return 0;
    case RIGHT: return 0;
    case UP: return 1;
    case LEFT: return 0;
    case DOWN: return -1;
    case UP_RIGHT: return 1;
    case UP_LEFT: return 1;
    case DOWN_LEFT: return -1;
    case DOWN_RIGHT: return -1;
  }
  return 0;
}

KOKKOS_FORCEINLINE_FUNCTION
double weight(Direction direction) {
  switch (direction) {
    case NONE: return 4. / 9.;
    case RIGHT: return 1. / 9.;
    case UP: return 1. / 9.;
    case LEFT: return 1. / 9.;
    case DOWN: return 1. / 9.;
    case UP_RIGHT: return 1. / 36.;
    case UP_LEFT: return 1. / 36.;
    case DOWN_LEFT: return 1. / 36.;
    case DOWN_RIGHT: return 1. / 36.;
  }
  return 0;
}

KOKKOS_FORCEINLINE_FUNCTION
Kokkos::pair<int, int> updated_coords(const int &x, const int &y, const Direction &dir, const unsigned int size_x, const unsigned int size_y) {
  int sx = static_cast<int>(size_x);
  int sy = static_cast<int>(size_y);
  int new_x = x + x_part(dir);
  int new_y = y + y_part(dir);
  new_x -= (new_x >= sx) * sx;
  new_x += (new_x < 0)   * sx;
  new_y -= (new_y >= sy) * sy;
  new_y += (new_y < 0)   * sy;
  return Kokkos::make_pair(new_x, new_y);
}
