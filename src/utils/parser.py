import re

class TenorParser:
    """ Utility class to parse tenors and convert into year fractions w.r.t. ACT/360"""
    DAY_COUNT = 360 # ACT/360

    @staticmethod
    def tenors_to_years(tenor: str) -> float:
        """ Converting tenor strings into year fractions """

        tenor = tenor.upper()

        if tenor == 'ON':
            return 1 / TenorParser.DAY_COUNT
        
        pattern = r'(\d+)([DWMY])'

        match = re.match(
            pattern = pattern,
            string = tenor
        )

        if not match:
            raise ValueError(f'Invalid tenor format: {tenor}')
        
        value, unit = match.groups()

        value = int(value)

        if unit == 'D':
            return value / TenorParser.DAY_COUNT
        
        elif unit == 'W':
            return value * 7 / TenorParser.DAY_COUNT

        elif unit == 'M':
            return value * 30 / TenorParser.DAY_COUNT

        elif unit == 'Y':
            return float(value)

        else:
            raise ValueError(f'Unsupported tenor unit: {unit}')