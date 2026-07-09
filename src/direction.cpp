#include "direction.hpp"
#include <cassert>
#include <tuple>

#define UNIT 1
const double y_part(Direction direction) {
  switch (direction) {
  case UP:
  case UP_RIGHT:
  case UP_LEFT:
    return UNIT;
  case DOWN:
  case DOWN_LEFT:
  case DOWN_RIGHT:
    return -UNIT;
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
    return UNIT;
  case LEFT:
  case DOWN_LEFT:
  case UP_LEFT:
    return -UNIT;
  case NONE:
  case UP:
  case DOWN:
  default:
    return 0;
  }
}

std::tuple<int, int> updated_coords(const int &x, const int &y, const int &dir, const unsigned int size_x, const unsigned int size_y) {
  int new_x;
  int new_y;
  switch (dir)
  {
    case NONE:
    case UP:
    case DOWN:
    default:
      new_x = x;
      break;
    case LEFT:
    case UP_LEFT:
    case DOWN_LEFT:
      new_x = x == 0 ? size_x - UNIT : x - UNIT;
      break;
    case RIGHT:
    case DOWN_RIGHT:
    case UP_RIGHT:
      new_x = x == size_x - UNIT ? 0 : x + UNIT;
      break;
  }
  switch (dir)
  {
    case NONE:
    case LEFT:
    case RIGHT:
    default:
      new_y = y;
      break;
    case UP:
    case UP_LEFT:
    case UP_RIGHT:
      new_y = y == size_y - UNIT ? 0 : y + UNIT;
      break;
    case DOWN:
    case DOWN_LEFT:
    case DOWN_RIGHT:
      new_y = y ==  0 ? size_y - UNIT : y - UNIT;
      break;
  }
  return std::tuple(new_x, new_y);
}

const double weight(Direction direction) {
  double num = direction != 0 ? 1 : 4;
  double denom = direction <= 4 ? 9 : 36;
  return num / denom;
}

