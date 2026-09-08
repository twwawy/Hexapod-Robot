"""MID-360 angular envelope shared by simulated returns and FOV visualization.

Source: https://www.livoxtech.com/mid-360/specs
The angular proxy does not reproduce Livox's non-repetitive scan or reflectivity.
"""
HORIZONTAL_FOV_DEG = 360.0
VERTICAL_FOV_DEG = (-7.0, 52.0)
MIN_RANGE_M = 0.1
LOW_DENSITY_SPLIT_DEG = 8.0
SPEC_URL = 'https://www.livoxtech.com/mid-360/specs'


def metadata():
    return dict(horizontal_deg=HORIZONTAL_FOV_DEG, vertical_deg=VERTICAL_FOV_DEG,
                min_range_m=MIN_RANGE_M, source=SPEC_URL,
                pattern='angular proxy, not Livox non-repetitive scan')
