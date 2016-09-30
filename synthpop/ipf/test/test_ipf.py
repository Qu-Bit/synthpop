import pandas as pd
import pytest
from pandas.util import testing as pdt

from .. import ipf


def test_trivial_ipf():
    # Test IPF in a situation where the desired totals and observed
    # sample have the same proportion and there is only one super-category.
    midx = pd.MultiIndex.from_product([('cat_owner',), ('yes', 'no')])
    marginals = pd.Series([60, 40], index=midx)
    joint_dist = pd.Series(
        [6, 4], index=pd.Series(['yes', 'no'], name='cat_owner'))

    expected = pd.Series(marginals.values, index=joint_dist.index)
    constraints, iterations = ipf.calculate_constraints(marginals, joint_dist)

    pdt.assert_series_equal(constraints, expected, check_dtype=False)
    assert iterations == 2


def test_larger_ipf():
    # Test IPF with some data that's slightly more meaningful,
    # but for which it's harder to know the actual correct answer.
    marginal_midx = pd.MultiIndex.from_tuples(
        [('cat_owner', 'yes'),
         ('cat_owner', 'no'),
         ('car_color', 'blue'),
         ('car_color', 'red'),
         ('car_color', 'green')])
    marginals = pd.Series([60, 40, 50, 30, 20], index=marginal_midx)
    joint_dist_midx = pd.MultiIndex.from_product(
        [('yes', 'no'), ('blue', 'red', 'green')],
        names=['cat_owner', 'car_color'])
    joint_dist = pd.Series([8, 4, 2, 5, 3, 2], index=joint_dist_midx)

    expected = pd.Series(
        [31.78776824, 17.77758309, 10.43464846,
         18.21223176, 12.22241691, 9.56535154],
        index=joint_dist.index)
    constraints, _ = ipf.calculate_constraints(marginals, joint_dist)

    pdt.assert_series_equal(constraints, expected, check_dtype=False)

    with pytest.raises(RuntimeError):
        ipf.calculate_constraints(marginals, joint_dist, max_iterations=2)


def test_not_add_ipf():
    # Test IPF with some data that's slightly more meaningful,
    # but for which it's harder to know the actual correct answer.
    marginal_midx = pd.MultiIndex.from_tuples(
        [('cat_owner', 'yes'),
         ('cat_owner', 'no'),
         ('car_color', 'blue'),
         ('car_color', 'red'),
         ('car_color', 'green')])
    marginals = pd.Series([60, 40, 50, 31, 20], index=marginal_midx)
    joint_dist_midx = pd.MultiIndex.from_product(
        [('yes', 'no'), ('blue', 'red', 'green')],
        names=['cat_owner', 'car_color'])
    joint_dist = pd.Series([8, 4, 2, 5, 3, 2], index=joint_dist_midx)

    with pytest.raises(RuntimeError):
        ipf.calculate_constraints(marginals, joint_dist)

def test_ipf_4OD():
    """test IPF for simple OD (2x2) cases
    """
    marginal_midx = pd.MultiIndex.from_tuples(
            [('o', 1),
             ('o', 2),
             ('d', 1),
             ('d', 2)])
    # exact sum marginals
    marginals = pd.Series([5, 6, 4, 7], index=marginal_midx)
    joint_dist_midx = pd.MultiIndex.from_product(
            [(1, 2), (1, 2)],
            names=['o', 'd'])
    joint_dist = pd.Series([3, 2, 1, 5], index=joint_dist_midx)
    # trivial case: identity
    constraints, _ = ipf.calculate_constraints(marginals, joint_dist)
    pdt.assert_series_equal(constraints, joint_dist, check_dtype=False)
    # with twice the marginals
    constraints, _ = ipf.calculate_constraints(marginals*2, joint_dist)
    pdt.assert_series_equal(constraints, joint_dist*2, check_dtype=False)

    # non-linear in d-sums
    marginals = pd.Series([5, 6, 3, 8], index=marginal_midx)
    constraints, _ = ipf.calculate_constraints(marginals, joint_dist)
    expected =  pd.Series([2.360679, 2.639257, 0.639321, 5.360743],
                          index=joint_dist.index)
    pdt.assert_series_equal(constraints, expected, check_dtype=False)

    # non-linear in o-sums
    marginals = pd.Series([7, 4, 4, 7], index=marginal_midx)
    constraints, _ = ipf.calculate_constraints(marginals, joint_dist)
    expected =  pd.Series([3.523743, 3.476232, 0.476257, 3.523768],
                          index=joint_dist.index)
    pdt.assert_series_equal(constraints, expected, check_dtype=False)


