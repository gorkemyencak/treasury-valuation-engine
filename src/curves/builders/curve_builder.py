from src.curves.curve_snapshot import CurveSnapshot

from src.curves.base_curve import BaseCurve

class CurveBuilder:
    """ Generic curve builder converting market curve snapshots into continuous curve representation """
    def __init__(
            self,
            interpolation_method: str = 'linear'
    ):
        # attributes
        self.interpolation_method = interpolation_method

        
    def build_curve(
            self,
            snapshot: CurveSnapshot
    ):
        """ Build continuous curve from curve snapshot """
        # validate the curve first before building BaseCurve object
        self._validate_snapshot(snapshot = snapshot)

        curve = BaseCurve(
            curve_snapshot = snapshot,
            interpolation_method = self.interpolation_method
        )

        return curve
    

    def _validate_snapshot(
            self,
            snapshot: CurveSnapshot
    ):
        """ Curve validation """
        if len(snapshot.tenors) != len(snapshot.rates):
            raise ValueError('Tenor & Rate length mismatch')
        
        if any(snapshot.tenor_years <= 0):
            raise ValueError('Invalid tenor year fractions')