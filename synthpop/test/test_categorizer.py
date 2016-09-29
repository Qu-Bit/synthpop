import pytest
import numpy as np
try:
    from ..census_helpers import Census
except ImportError:
    pass
from .. import categorizer as cat

def got_module(mod_name):
    return mod_name in dir()


# TODO: keep these tests,
#       but REMOVE the census deps by using static test data

@pytest.mark.skipif(not got_module('census'), reason='census module not available')
@pytest.fixture
def c():
    return Census("827402c2958dcf515e4480b7b2bb93d1025f9389")


@pytest.mark.skipif(not got_module('census'), reason='census module not available')
@pytest.fixture
def acs_data(c):
    population = ['B01001_001E']
    sex = ['B01001_002E', 'B01001_026E']
    race = ['B02001_0%02dE' % i for i in range(1, 11)]
    male_age_columns = ['B01001_0%02dE' % i for i in range(3, 26)]
    female_age_columns = ['B01001_0%02dE' % i for i in range(27, 50)]
    all_columns = population + sex + race + male_age_columns + \
        female_age_columns
    df = c.block_group_query(all_columns, "06", "075", tract="030600")
    return df


@pytest.mark.skipif(not got_module('census'), reason='census module not available')
@pytest.fixture
def pums_data(c):
    return c.download_population_pums("06", "07506")


@pytest.mark.skipif(not got_module('census'), reason='census module not available')
def test_categorize(acs_data, pums_data):
    p_acs_cat = cat.categorize(acs_data, {
        ("population", "total"): "B01001_001E",
        ("age", "19 and under"): "B01001_003E + B01001_004E + B01001_005E + "
                                 "B01001_006E + B01001_007E + B01001_027E + "
                                 "B01001_028E + B01001_029E + B01001_030E + "
                                 "B01001_031E",
        ("age", "20 to 35"): "B01001_008E + B01001_009E + B01001_010E + "
                             "B01001_011E + B01001_012E + B01001_032E + "
                             "B01001_033E + B01001_034E + B01001_035E + "
                             "B01001_036E",
        ("age", "35 to 60"): "B01001_013E + B01001_014E + B01001_015E + "
                             "B01001_016E + B01001_017E + B01001_037E + "
                             "B01001_038E + B01001_039E + B01001_040E + "
                             "B01001_041E",
        ("age", "above 60"): "B01001_018E + B01001_019E + B01001_020E + "
                             "B01001_021E + B01001_022E + B01001_023E + "
                             "B01001_024E + B01001_025E + B01001_042E + "
                             "B01001_043E + B01001_044E + B01001_045E + "
                             "B01001_046E + B01001_047E + B01001_048E + "
                             "B01001_049E",
        ("race", "white"):   "B02001_002E",
        ("race", "black"):   "B02001_003E",
        ("race", "asian"):   "B02001_005E",
        ("race", "other"):   "B02001_004E + B02001_006E + B02001_007E + "
                             "B02001_008E",
        ("sex", "male"):     "B01001_002E",
        ("sex", "female"):   "B01001_026E"
    }, index_cols=['NAME'])

    assert len(p_acs_cat) == 3
    assert len(p_acs_cat.columns) == 11
    assert len(p_acs_cat.columns.names) == 2
    assert p_acs_cat.columns[0][0] == "age"

    assert np.all(cat.sum_accross_category(p_acs_cat) < 2)

    def age_cat(r):
        if r.AGEP <= 19:
            return "19 and under"
        elif r.AGEP <= 35:
            return "20 to 35"
        elif r.AGEP <= 60:
            return "35 to 60"
        return "above 60"

    def race_cat(r):
        if r.RAC1P == 1:
            return "white"
        elif r.RAC1P == 2:
            return "black"
        elif r.RAC1P == 6:
            return "asian"
        return "other"

    def sex_cat(r):
        if r.SEX == 1:
            return "male"
        return "female"

    pums_data, jd_persons = cat.joint_distribution(
        pums_data,
        cat.category_combinations(p_acs_cat.columns),
        {"age": age_cat, "race": race_cat, "sex": sex_cat}
    )
