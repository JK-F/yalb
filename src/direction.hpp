#include <tuple>
#define NUM_DIRECTIONS 9

enum Direction {
  // No Y movement
  NONE = 0,
  RIGHT = 1,
  LEFT = 3,
  // No X Movement
  UP = 2,
  DOWN = 4,
  // Diagonals
  UP_RIGHT = 5,
  UP_LEFT = 6,
  DOWN_LEFT = 7,
  DOWN_RIGHT = 8,
};

double y_part(Direction direction);
double x_part(Direction direction);
double weight(Direction direction);

std::tuple<int, int> updated_coords(const int &x, const int &y, const Direction &dir, const unsigned int size_x, const unsigned int size_y);
