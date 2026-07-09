#include "direction.hpp"
#include <cassert>
#include <tuple>


// Look up tables
static constexpr double x_lut[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static constexpr double y_lut[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static constexpr double w_lut[9] = {
                                    /*NONE*/        4. / 09.,
                                    /*L,R,U,D*/     1. / 09., 1. / 09., 1. / 09., 1. / 09.,
                                    /*Diagonal*/    1. / 36., 1. / 36., 1. / 36., 1. / 36., 
                                    };

const double x_part(Direction direction) {
  return x_lut[direction];
}

const double y_part(Direction direction) {
  return y_lut[direction];
}

std::tuple<int, int> updated_coords(const int &x, const int &y, const Direction &dir, const unsigned int size_x, const unsigned int size_y) {
  int sx = static_cast<int>(size_x);
  int sy = static_cast<int>(size_x);
  int new_x = x + x_part(dir);;
  int new_y = y + y_part(dir);;
  // Boolean arithmentic magic
  // Suggested by Deepseek
  // It's actually clever
  new_x -= (new_x >= sx) * sx;   // wraps sx → 0 <--- If new_x < sx it stays, otherwise (new_x = -1) we add sx => -1 + sx  
  new_x += (new_x < 0)   * sx;   
  new_y -= (new_y >= sy) * sy;
  new_y += (new_y < 0)   * sy;
  return std::tuple(new_x, new_y);
}

const double weight(Direction direction) {
  return w_lut[direction];
}

