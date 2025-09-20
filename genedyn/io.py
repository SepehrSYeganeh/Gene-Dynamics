import os
import csv
import pandas as pd
from multiprocessing import cpu_count


def init_data(index: int) -> None:
    """initialize data files"""
    FILENAME = f"data/data{index}.csv"
    HEADER = ['dynamics',
              'cc', 'rc', 'hc', 'cs', 'ss', 'hs',
              'mymy', 'hmy', 'memy', 'rr', 'hr', 'rh',
              'hh', 'meh', 'hme', 'meme', 'cr', 'sr', 'myh',
              'CEBPA', 'SPI1', 'MYB', 'RUNX1', 'HOXA9', 'MEIS1']
    os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
    with open(FILENAME, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)


def append_data(index: int, params: dict, dynamics_type: str, fixed_points: list) -> None:
    """append data"""
    FILENAME = f"data/data{index}.csv"
    data = [dynamics_type,
            params['cc'], params['rc'], params['hc'], params['cs'], params['ss'], params['hs'],
            params['mymy'], params['hmy'], params['memy'], params['rr'], params['hr'], params['rh'],
            params['hh'], params['meh'], params['hme'], params['meme'], params['cr'], params['sr'], params['myh'],
            fixed_points[0], fixed_points[1], fixed_points[2], fixed_points[3], fixed_points[4], fixed_points[5]]
    with open(FILENAME, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(data)


def load_data() -> pd.DataFrame:
    """load all data files in one dataframe"""
    dfs = [pd.read_csv(f"data/data{i}.csv") for i in range(cpu_count())]
    return pd.concat(dfs, ignore_index=True)
