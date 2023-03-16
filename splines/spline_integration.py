from typing import List

from regression_spline_estimator import Regression_Spline_Estimator


def integrate_spline(spline_list: List[Regression_Spline_Estimator], order: int):
    """Computes the kth integral of a spline returning a new spline object"""


    ### integrate product of splines


    knot_sets = [spline.basis.gamma for spline in spline_list]
    assert set(knot_sets) == 1, "all knot sets must be the same"

    # extract pairs of knots
    knot_pairs = [(knot_sets[0][i],knot_sets[0][i+1]) for i in range(len(knot_sets[0])-1) ]

    # compute integral for product of splines across each subinterval
    for i in range(len(knot_pairs)):
        