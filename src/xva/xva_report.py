import pandas as pd

from src.xva.cva_engine import CVAEngine
from src.xva.dva_engine import DVAEngine

class XVAReport:
    """ Reporting layer for key components of xVA """
    def __init__(
            self,
            cva_engine: CVAEngine,
            dva_engine: DVAEngine

    ):
        # attributes
        self.cva_engine = cva_engine
        self.dva_engine = dva_engine

    def summary_report(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ Summary report for xVA components """
        # xVA components
        _, cva = self.cva_engine.cva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        _, dva = self.dva_engine.dva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        bva = dva - cva

        return pd.DataFrame({
            'Metrics': [
                'CVA',
                'DVA',
                'BVA'
            ],
            'Value': [
                cva,
                dva,
                bva
            ]
        })
    
    def full_report(
            self,
            swap,
            n_paths: int = 250,
            steps_per_year: int = 4
    ):
        """ Full comprehensive report for xVA components """
        # xVA components
        cva_profile, cva = self.cva_engine.cva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        dva_profile, dva = self.dva_engine.dva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        report = (
            cva_profile[['Times', 'CVA_Contribution']]
            .merge(dva_profile[['Times', 'DVA_Contribution']], on = 'Times')
        )

        report['BVA_Contribution'] = report['DVA_Contribution'] - report['CVA_Contribution']

        return report