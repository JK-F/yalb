#include <Kokkos_Core.hpp>
#include <utility>
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

KOKKOS_INLINE_FUNCTION
double x_part(Direction direction) {
  static constexpr double lut[] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
  return lut[direction];
}

KOKKOS_INLINE_FUNCTION
double y_part(Direction direction) {
  static constexpr double lut[] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
  return lut[direction];
}

KOKKOS_INLINE_FUNCTION
double weight(Direction direction) {
  static constexpr double lut[] = {4. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 9., 1. / 36., 1. / 36., 1. / 36., 1. / 36.};
  return lut[direction];
}

KOKKOS_INLINE_FUNCTION
std::pair<int, int> updated_coords(const int &x, const int &y, const Direction &dir, const unsigned int size_x, const unsigned int size_y) {
  int sx = static_cast<int>(size_x);
  int sy = static_cast<int>(size_y);
  int new_x = x + x_part(dir);
  int new_y = y + y_part(dir);
  new_x -= (new_x >= sx) * sx;
  new_x += (new_x < 0)   * sx;
  new_y -= (new_y >= sy) * sy;
  new_y += (new_y < 0)   * sy;
  return std::make_pair(new_x, new_y);
}
