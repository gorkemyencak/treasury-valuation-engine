# US market curves
FRED_CONFIG = {
    # risk-free proxies
    'treasury': {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '5Y': 'DGS5',
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    },
    # derivatives discounting curve
    'sofr': {
        'ON': 'SOFR',
        'FEDFUNDS': 'DFF'
    },
    # future proxies
    'futures': {
        'TBill3M': 'DTB3',
        'TBill6M': 'DTB6'
    }
}

# Euribor proxies and short-term rates
ECB_CONFIG = {
    'estr': {
        'ESTR': 'EST.B.EU000A2X2A25.WT' # Euro short-term rate
    }
}

# DTCC Swap rates
DTCC_CONFIG = {
    'usd_ois': {
        'currency': 'USD',
        'index': 'SOFR',
        'instrument_type': 'OIS',
        'source': 'DTCC',
        'tenors': [
            '1Y',
            '2Y',
            '3Y',
            '4Y',
            '5Y',
            '7Y',
            '10Y',
            '15Y',
            '20Y',
            '30Y'
        ]
    },
    'usd_irs': {
        'currency': 'USD',
        'index': 'SOFR',
        'instrument_type': 'IRS',
        'source': 'DTCC',
        'tenors': [
            '1Y',
            '2Y',
            '3Y',
            '4Y',
            '5Y',
            '7Y',
            '10Y',
            '15Y',
            '20Y',
            '30Y'
        ]
    }
}