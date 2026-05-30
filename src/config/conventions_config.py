# market conventions for different indices
MARKET_CONVENTIONS = {
    'usd_ois': {
        'currency': 'USD',
        'day_count': 'ACT/360',
        'payment_frequency': '1Y'
    },
    'usd_irs': {
        'currency': 'USD',
        'fixed_leg_frequency': '6M',
        'floating_leg_frequency': '3M',
        'day_count': '30/360'
    }
}

CURVE_BUILD = {
    'usd_discount_curve': {
        'short_end': 'sofr',
        'middle': 'futures',
        'long_end': 'usd_ois'
    },
    'usd_projection_curve': {
        'short_end': 'sofr',
        'middle': 'futures',
        'long_end': 'usd_irs'
    }    
}