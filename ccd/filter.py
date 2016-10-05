"""Filters for pre-processing change model inputs.
"""


def enough_clear(qa):
    """ some % of clear vs not clear """
    pass


def calculate_percent_snow(observations, qa):
    """ sn_pct = sum(idsn)/sum(idclr) + sum(idsn) + 0.01 """
    pass


def calculate_percent_clear(qa):
    """ sum(idclr)/sum(idall) """
    pass


def enough_snow(qa):
    """ sn_pct > t_sn """
    pass


def are_saturated(pixels):
    """ """
    pass


def filter_out_of_range_specta(spectrum):
    """ 0 to 10000 for all bands but thermal.  Thermal is -93.2C to 70.7C
    179.95K -- 343.85K """
    pass


def is_fill():
    pass


def is_cloud():
    pass


def is_snow():
    pass


def is_water():
    pass


def is_clear():
    pass


def categorize(qa):
    """ determine the category to use for detecting change """
    """
    IF clear_pct IS LESS THAN CLEAR_OBSERVATION_THRESHOLD
    THEN

        IF permanent_snow_pct IS GREATER THAN PERMANENT_SNOW_THRESHOLD
        THEN
            IF ENOUGH snow pixels
            THEN
                DO snow based change detection
            ELSE
                BAIL

    ELSE
        DO NORMAL CHANGE DETECTION


    """


def preprocess(matrix):
    pass
