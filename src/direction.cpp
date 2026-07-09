#include "direction.hpp"
#include <cassert>
#include <iterator>
#include <tuple>

const double y_part(Direction direction) {
  switch (direction) {
  case UP:
  case UP_RIGHT:
  case UP_LEFT:
    return 1;
  case DOWN:
  case DOWN_LEFT:
  case DOWN_RIGHT:
    return -1;
  case NONE:
  case RIGHT:
  case LEFT:
  default:
    return 0;
  }
}

const double x_part(Direction direction) {
  switch (direction) {
  case RIGHT:
  case DOWN_RIGHT:
  case UP_RIGHT:
    return 1;
  case LEFT:
  case DOWN_LEFT:
  case UP_LEFT:
    return -1;
  case NONE:
  case UP:
  case DOWN:
  default:
    return 0;
  }
}

std::tuple<int, int> updated_coords(const int &x, const int &y, const Direction &dir, const unsigned int size_x, const unsigned int size_y) {
  int dx = x_part(dir);
  int dy = y_part(dir);
  int new_x = x + dx;
  int new_y = y + dy;
  if (new_x >= size_x) {
    new_x = 0;
  }
  else if (new_x < 0) {
    new_x = size_x - 1;
  }

  if (new_y >= size_y) {
    new_y = 0;
  }
  else if (dy < 0) {
    new_y = size_y - 1;
  }
  return std::tuple(new_x, new_y);
}

const double weight(Direction direction) {
  double num = direction != 0 ? 1 : 4;
  double denom = direction <= 4 ? 9 : 36;
  return num / denom;
}

