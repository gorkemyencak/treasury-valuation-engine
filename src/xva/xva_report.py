import pandas as pd

from src.xva.cva_engine import CVAEngine
from src.xva.dva_engine import DVAEngine
from src.xva.fva_engine import FVAEngine
from src.xva.colva_engine import ColVAEngine
from src.xva.mva_engine import MVAEngine
from src.xva.kva_engine import KVAEngine

class XVAReport:
    """ Reporting layer for key components of xVA """
    def __init__(
            self,
            cva_engine: CVAEngine,
            dva_engine: DVAEngine,
            fva_engine: FVAEngine,
            colva_engine: ColVAEngine,
            mva_engine: MVAEngine,
            kva_engine: KVAEngine
    ):
        # attributes
        self.cva_engine = cva_engine
        self.dva_engine = dva_engine
        self.fva_engine = fva_engine
        self.colva_engine = colva_engine
        self.mva_engine = mva_engine
        self.kva_engine = kva_engine

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

        _, fva = self.fva_engine.fva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        _, colva = self.colva_engine.colva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        _, mva = self.mva_engine.mva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        _, kva = self.kva_engine.kva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        bva = dva - cva

        xva = dva - cva - fva - colva - mva - kva

        return pd.DataFrame({
            'Metrics': [
                'CVA',
                'DVA',
                'BVA',
                'FVA',
                'ColVA',
                'MVA',
                'KVA',
                'XVA'
            ],
            'Value': [
                cva,
                dva,
                bva,
                fva,
                colva,
                mva,
                kva,
                xva
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

        fva_profile, fva = self.fva_engine.fva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year 
        )

        colva_profile, colva = self.colva_engine.colva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        mva_profile, mva = self.mva_engine.mva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        kva_profile, kva = self.kva_engine.kva_profile(
            swap = swap,
            n_paths = n_paths,
            steps_per_year = steps_per_year
        )

        report = (
            cva_profile[['Times', 'CVA_Contribution']]
            .merge(dva_profile[['Times', 'DVA_Contribution']], on = 'Times')
            .merge(fva_profile[['Times', 'FVA_Contribution']], on = 'Times')
            .merge(colva_profile[['Times', 'ColVA_Contribution']], on = 'Times')
            .merge(mva_profile[['Times', 'MVA_Contribution']], on = 'Times')
            .merge(kva_profile[['Times', 'KVA_Contribution']], on = 'Times')
        )

        report['BVA_Contribution'] = report['DVA_Contribution'] - report['CVA_Contribution']
        report['XVA_Contribution'] = (
            report['DVA_Contribution'] 
            - report['CVA_Contribution'] 
            - report['FVA_Contribution']
            - report['ColVA_Contribution']
            - report['MVA_Contribution']
            - report['KVA_Contribution']
        )
        return report