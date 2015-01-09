#!/usr/bin/env python
# coding: utf-8
import scipy as sp
import scipy.interpolate as spi
import numpy as np
import matplotlib
matplotlib.use('SVG')
import matplotlib.pyplot as plt
import random

HEART_LEFT_X = [1498, 1543, 1536, 1471, 1241, 890, 634, 566, 626, 819, 905, 1139, 1343, 1437]
HEART_LEFT_Y = [0, 197, 276, 438, 706, 932, 1200, 1442, 1684, 1895, 1925, 1936, 1842, 1744]
HEART_RIGHT_X = [1385, 1573, 1830, 2072, 2264, 2373, 2328, 2173, 1951, 1766, 1668, 1634, 1630]
HEART_RIGHT_Y = [1604, 1816, 1936, 1906, 1755, 1495, 1216, 1027, 849, 748, 653, 559, 491]

HEART_BOTH_X = HEART_LEFT_X + HEART_RIGHT_X
HEART_BOTH_Y = HEART_LEFT_Y + HEART_RIGHT_Y

HEART_LEFT_WIDTH = 140
HEART_RIGHT_WIDTH = 300

THICKNESS_PROFILE_X = range(11)
THICKNESS_PROFILE_Y = [0., 0.3, 0.8, 1., 0.9, 0.2, 0.2, 0.7, 0.7, 0.3, 0.]

# DEFAULT_COLOR_1 = '#86BBC9'
# DEFAULT_COLOR_2 = '#DF036F'
# DEFAULT_COLOR_3 = '#8DB401'
# DEFAULT_COLOR_4 = '#E6D106'

DEFAULT_COLOR_1 = '#00bbff'
DEFAULT_COLOR_2 = '#ff006f'
DEFAULT_COLOR_3 = '#8dff00'
DEFAULT_COLOR_4 = '#ffd100'

FLOWER_SWITCH = "FlowerPower"
DEFAULT_NUM_CIRCLES = 120
DEFAULT_ALPHA = 0.8
DEFAULT_EDGECOLOR = 'none'
DEFAULT_RAD_MIN = 5
DEFAULT_RAD_MAX = 50
DEFAULT_BG_COLOR = "white"
DEFAULT_FLOWER_CENTER_RADIUS_FRACTION = 4. / 11
DEFAULT_FLOWER_LEAF_INNER_RADIUS_FRACTION = 3. / 11
DEFAULT_FLOWER_LEAF_COUNT = 12
DEFAULT_FLOWER_WIDTH_FACTOR = 0.7
DEFAULT_FLOWER_LEAF_COLOR = "yellow"
DEFAULT_FLOWER_CENTER_COLOR = "white"
DEFAULT_FLOWER_CENTER_ALPHA = 1.0
DEFAULT_FLOWER_EDGECOLOR = 'none'


def smooth_function_transversal(x):
  return 1. * HEART_LEFT_WIDTH * (1. - np.exp(-0.5 * (3 * x) ** 2))


def smooth_function_longitudinal(x):
  abs_x = np.abs(x)
  val = 0.5 * np.e * np.exp(-1. / (1. - abs_x ** 2))
  if x < 0:
    return val
  else:
    return 1. - val


class Curve():
  num_points = 100

  def __init__(self, xar, yar, smoothness=None):
    self.smoothness = 100
    if smoothness is not None:
      self.smoothness = smoothness
    self.tck, self.u = spi.splprep([xar, yar], s=self.smoothness)
    unew = np.linspace(0., 1., self.num_points + 1)
    self.crv_coords = spi.splev(unew, self.tck)
    self.crv_der = spi.splev(unew, self.tck, der=1)

  def get_index(self, s):
    return int(s * self.num_points)

  def get_point(self, s):
    idx = self.get_index(s)
    ret_x = self.crv_coords[0][idx]
    ret_y = self.crv_coords[1][idx]
    return ret_x, ret_y

  def get_point_with_normal_shift(self, s, shift):
    ret_x, ret_y = self.get_point(s)
    nx, ny = self.get_normal(s)
    ret_x += nx * shift
    ret_y += ny * shift
    return ret_x, ret_y

  def get_tangent(self, s):
    idx = self.get_index(s)
    ret_x = self.crv_der[0][idx]
    ret_y = self.crv_der[1][idx]
    return ret_x, ret_y

  def get_normal(self, s):
    idx = self.get_index(s)
    ret_x = self.crv_der[1][idx]
    ret_y = -1. * self.crv_der[0][idx]
    nrm_len = np.sqrt(ret_x ** 2 + ret_y ** 2)
    return ret_x / nrm_len, ret_y / nrm_len

  def show_curve(self):
    plt.plot(self.crv_coords[0], self.crv_coords[1])
    for i in np.linspace(0.1, 0.9, 9):
      x, y = self.get_point(i)
      plt.plot(x, y, 'o')
    plt.gca().set_aspect('equal')
    plt.show()


class CircleList():
  rad_min = DEFAULT_RAD_MIN
  rad_max = DEFAULT_RAD_MAX

  def __init__(self, curve, sfl, sft):
    self.crv = curve
    self.sfl = sfl
    self.sft = sft
    self.circles = []

  def populate(self, num, color, non_overlapping=False):
    circles_copy = self.circles[:]
    self.circles = []
    while len(self.circles) < num:
      self.add_new_circle(color, non_overlapping)
    circles_copy.extend(self.circles)
    self.circles = circles_copy

  def add_new_circle(self, color, non_overlapping=False):
    pp, shift = self.get_values_for_new_circle()
    px, py = self.crv.get_point_with_normal_shift(pp, shift)
    radius = random.uniform(self.rad_min, self.rad_max)
    if not non_overlapping or not self.is_overlapping(px, py, radius):
      self.circles.append([px, py, radius, color])
      return True
    return False

  def get_values_for_new_circle(self):
    pathpoint = self.sfl(random.uniform(-1, 1))
    shift = self.sft(random.uniform(-1, 1))
    return pathpoint, shift

  def is_overlapping(self, px, py, radius, color=None):
    for [cx, cy, crad, ccol] in self.circles:
      if (color is None or color == ccol) and np.sqrt((cx - px) ** 2 + (cy - py) ** 2) < radius + crad:
        return True
    return False

  def get_minimal_frame_dimensions(self, equal_aspect=True):
    xmin = xmax = self.circles[0][0]
    ymin = ymax = self.circles[0][1]
    for [cx, cy, crad, ccol] in self.circles:
      xmin = min(cx - crad, xmin)
      xmax = max(cx + crad, xmax)
      ymin = min(cy - crad, ymin)
      ymax = max(cy + crad, ymax)
    if equal_aspect:
      w = xmax - xmin
      h = ymax - ymin
      if h > w:
        xmin -= (h - w) / 2
        xmax += (h - w) / 2
      else:
        ymin -= (w - h) / 2
        ymax += (w - h) / 2
    return [xmin, xmax, ymin, ymax]

  def show_plot(self, save_fig_filename=None):
    fig = plt.gcf()
    fig.set_facecolor(DEFAULT_BG_COLOR)
    for [px, py, radius, color] in self.circles:
      if color == FLOWER_SWITCH:
        cur_flower = Flower((px, py), radius)
        cur_flower.plot_on_axes(fig.gca())
      else:
        c = plt.Circle((px, py),
                       radius,
                       color=color,
                       alpha=DEFAULT_ALPHA)
        c.set_edgecolor(DEFAULT_EDGECOLOR)
        fig.gca().add_artist(c)
    plt.axis(self.get_minimal_frame_dimensions())
    plt.gca().set_axis_off()
    plt.gca().set_aspect('equal')
    if save_fig_filename is not None:
      plt.savefig(save_fig_filename, edgecolor=DEFAULT_FLOWER_EDGECOLOR, transparent=True)
    plt.show()


class Flower():
  def __init__(self, xy, radius, angle_shift=0):
    self.px, self.py = xy
    self.radius = radius
    self.angle_shift = angle_shift

  def plot_on_axes(self, axes):
    for i in range(DEFAULT_FLOWER_LEAF_COUNT):
      e = matplotlib.patches.Ellipse(
          self.get_leaf_center(i),
          self.get_leaf_height(),
          self.get_leaf_width(),
          self.get_current_angle(i),
          color=DEFAULT_FLOWER_LEAF_COLOR,
          alpha=DEFAULT_ALPHA)
      e.set_edgecolor(DEFAULT_FLOWER_EDGECOLOR)
      axes.add_artist(e)
    c = plt.Circle((self.px, self.py),
                   DEFAULT_FLOWER_CENTER_RADIUS_FRACTION * self.radius,
                   color=DEFAULT_FLOWER_CENTER_COLOR,
                   alpha=DEFAULT_FLOWER_CENTER_ALPHA)
    axes.add_artist(c)

  def get_current_angle(self, num_leaf):
    return self.angle_shift + num_leaf * 360 / DEFAULT_FLOWER_LEAF_COUNT

  def get_leaf_center(self, num_leaf):
    center_radius = self.get_radius_of_leaf_center()
    cur_angle = self.get_current_angle(num_leaf)
    sx = np.cos(2. * np.pi * cur_angle / 360) * center_radius
    sy = np.sin(2. * np.pi * cur_angle / 360) * center_radius
    return self.px + sx, self.py + sy

  def get_leaf_width(self):
    return DEFAULT_FLOWER_WIDTH_FACTOR * 2. * np.pi * self.get_radius_of_leaf_center() / DEFAULT_FLOWER_LEAF_COUNT

  def get_radius_of_leaf_center(self):
    return (1. + DEFAULT_FLOWER_LEAF_INNER_RADIUS_FRACTION) / 2 * self.radius

  def get_leaf_height(self):
    return self.radius * (1. - DEFAULT_FLOWER_LEAF_INNER_RADIUS_FRACTION)


class CircleListWithThicknessProfile(CircleList):
  def __init__(self, curve, thickness_profile):
    self.crv = curve
    self.profile = thickness_profile
    self.circles = []

  def get_values_for_new_circle(self):
    pathpoint = random.random()
    discard, max_shift = self.profile.get_point(pathpoint)
    shift = random.uniform(0., max_shift)
    shift *= HEART_RIGHT_WIDTH
    return pathpoint, shift


def main():
  # hl = Curve(HEART_LEFT_X, HEART_LEFT_Y)
  hl = Curve(HEART_BOTH_X, HEART_BOTH_Y)
  # cl = CircleList(hl, smooth_function_longitudinal, smooth_function_transversal)
  tp = Curve(THICKNESS_PROFILE_X, THICKNESS_PROFILE_Y, 0.01)
  cl = CircleListWithThicknessProfile(hl, tp)

  # cl.populate(DEFAULT_NUM_CIRCLES, '#CBDB5E', True)

  cl.populate(DEFAULT_NUM_CIRCLES, DEFAULT_COLOR_1, True)
  cl.populate(DEFAULT_NUM_CIRCLES, DEFAULT_COLOR_2, True)
  cl.populate(DEFAULT_NUM_CIRCLES, DEFAULT_COLOR_3, True)
  cl.populate(DEFAULT_NUM_CIRCLES, DEFAULT_COLOR_4, True)
  cl.populate(DEFAULT_NUM_CIRCLES, FLOWER_SWITCH, False)
  cl.show_plot('heart.svg')
  # cl.show_plot()

if __name__ == '__main__':
  main()
