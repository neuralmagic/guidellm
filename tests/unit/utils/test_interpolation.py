import pytest
from guidellm.utils.interpolation import (
  linear_interpolate,
  interpolate_measurements,
  interpolate_data_points,
  stretch_list
)

@pytest.mark.smoke()
def test_linear_interpolate():
  assert linear_interpolate(2, (1, 4), (3, 6)) == 5
  

@pytest.mark.smoke()
def test_stretch_list():
  assert stretch_list([1,3,5], 5) == [1,2,3,4,5]
  
@pytest.mark.smoke()
def test_interpolate_measurements():
  assert interpolate_measurements(2, (1, [1,2,3,4,5]), (3, [2,3,4,5,6])) == [1.5,2.5,3.5,4.5,5.5]
  assert interpolate_measurements(2, (1, [1,2,3,4,5]), (3, [5,4,3,2,1])) == [3,3,3,3,3]

  
@pytest.mark.smoke()
def test_interpolate_data_point():
  assert interpolate_data_points([(1, [1,2,3,4,5]), (3, [2,3,4,5,6]), (9, [5,6,7,8,9])], [1,2,3,4,5,6,7,8,9]) == [(1, [1,2,3,4,5]), (2, [1.5,2.5,3.5,4.5,5.5]), (3, [2,3,4,5,6]), (4, [2.5, 3.5, 4.5, 5.5, 6.5]), (5, [3, 4, 5, 6, 7]), (6, [3.5, 4.5, 5.5, 6.5, 7.5]), (7, [4, 5, 6, 7, 8]), (8, [4.5, 5.5, 6.5, 7.5, 8.5]), (9, [5, 6, 7, 8, 9])]