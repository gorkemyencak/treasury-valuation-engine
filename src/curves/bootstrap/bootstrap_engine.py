from src.curves.discount_curve import DiscountCurve

class BootstrapCurveEngine:
    """ Bootstrap discount curve from market instruments """
    def __init__(self):
        pass


    def bootstrap(
            self,
            snapshot,
            instruments,
            interpolation_method = 'linear'
    ):
        # validatng non-empty instruments
        if len(instruments) <= 0:
            raise ValueError('No instruments supplied!')

        # ensure sorted instruments
        instruments = sorted(
            instruments,
            key = lambda x: x.maturity
        )

        discount_factor_map = {}
        
        maturities = []
        dfs = []

        for inst in instruments:

            maturity = inst.maturity

            df = inst.implied_discount_factor(
                curve = discount_factor_map
            )

            discount_factor_map[maturity] = df

            maturities.append(maturity)
            dfs.append(df)
        
        return DiscountCurve(
            curve_snapshot = snapshot,
            maturities = maturities,
            discount_factors = dfs,
            interpolation_method = interpolation_method
        )