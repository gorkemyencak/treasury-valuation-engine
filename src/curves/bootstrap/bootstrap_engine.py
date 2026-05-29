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
        
        maturities = []
        dfs = []

        for inst in instruments:

            maturity = inst.maturity

            df = inst.implied_discount_factor()

            maturities.append(maturity)
            dfs.append(df)
        
        return DiscountCurve(
            curve_snapshot = snapshot,
            maturities = maturities,
            discount_factors = dfs,
            interpolation_method = interpolation_method
        )