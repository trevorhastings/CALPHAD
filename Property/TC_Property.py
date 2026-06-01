import numpy as np
import pandas as pd
from tc_python import TCPython
from tc_python import CompositionUnit
import concurrent.futures
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import functools
import time
from typing import Protocol

from thermocalc_library import check_division, check_temperature_limit, check_single_system
from thermocalc_library import write_cpu_log, collect_cpu_logs, write_execution_log
from thermocalc_library import collect_version_info
from thermocalc_library import divide_up_cores
from thermocalc_library import create_parallelization_list
from thermocalc_library import concatenate_divided_data


class PropertyResultLike(Protocol):
    '''TC Python internals may change without warning.
    If the instance of the TC class has any of the below methods, then that assigned object
    will pass static type checkers.'''
    def get_value_of(self, name: str) -> float: ...

def Property(
        database: str,
        upper_temp_limit_in_kelvin: int,
        folder: str,
        elements: list[str],
        model_input: pd.DataFrame) -> None:
    t0 = time.monotonic()
    with TCPython() as session:
        pr_calculation: PropertyResultLike = (session.select_database_and_elements(
            database, elements)
                          .get_system()
                          .with_property_model_calculation("Liquidus and Solidus Temperature")
                          .set_argument('upperTemperatureLimit', upper_temp_limit_in_kelvin)
                          .set_argument('maxNumberOfIterations', 10)
                          .set_composition_unit(CompositionUnit.MOLE_FRACTION))
        cpu_block: list[int] = list(model_input.index)
        col_1: str = 'Liquidus_temperature_(C)'
        col_2: str = 'Solidus_temperature_(C)'
        for i in model_input.index:
            comp: np.ndarray = np.array(model_input.loc[i][elements])
            for j in range(len(elements) - 1):
                pr_calculation: PropertyResultLike = pr_calculation.set_composition(
                    elements[j], comp[j])
            try:
                result: PropertyResultLike = pr_calculation.calculate()
                T_liquidus_K = result.get_value_of('Liquidus temperature')
                T_solidus_K = result.get_value_of('Solidus temperature')
                model_input.loc[i, col_1] = T_liquidus_K - 273.15
                model_input.loc[i, col_2] = T_solidus_K - 273.15
            except Exception as e:
                print(f'Error in calculation for index {i}: {e}')
            finally:
                model_input.to_csv(f'{folder}/PROP_OUT_{model_input.index[0]}.csv', index = False)
        # ---- End CPU loop through list
        print(f'Completed: PROP_OUT_{model_input.index[0]}.csv')
    # ---- End TCPython Session
    elapsed_time_model: float = time.monotonic() - t0
    write_cpu_log(folder, model_input, elapsed_time_model, cpu_block)
    return None

if __name__ == '__main__':
    
    t1_1: float = time.monotonic()
    
    # ---- User inputted parameters
    file_name: str = 'TiVCo_n3_d25_single_system'
    file_divisions: int = 12
    cpu_cores: int = 12
    version: str = '2024b'
    database = 'TCHEA7'
    upper_temp_limit_in_kelvin: int = 2600 # e.g. 3000

    folder = f'{file_name}_TCprop2'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    data_input: pd.DataFrame = pd.read_csv(f"{file_name}.csv")
    # ---- Perform checks
    check_division: bool = check_division(file_divisions, cpu_cores)
    check_single_system: bool = check_single_system(data_input)
    check_temperature_limit(data_input, upper_temp_limit_in_kelvin)
    make_database_inquiry: bool = True # Only set this to False if you are running the script *locally*
    version_info: list[str]
    check_database: bool
    version_info, check_database = collect_version_info(version, database, make_database_inquiry)
    with open(os.path.join(folder, "all_checks_passed.txt"), "w", newline="") as f:
        pass
    # ---- Set up divisions
    elements: list[str] = [col for col in data_input.columns if len(col) <= 2 and data_input.iloc[0][col] > 0]
    rows_per_division, rows_total = divide_up_cores(data_input, file_divisions)
    model_inputs = create_parallelization_list(data_input, rows_per_division)
    partial_property = functools.partial(
        Property,
        database,
        upper_temp_limit_in_kelvin,
        folder,
        elements)
    # ---- Worker processes
    with concurrent.futures.ProcessPoolExecutor(cpu_cores) as executor:
        for _ in executor.map(partial_property, model_inputs):
            pass
        
    concatenate_divided_data(folder = folder, file_start = 'PROP_OUT_')

    cpu_logs: list[str] = collect_cpu_logs(folder)
    
    t1_2: float = time.monotonic()
    elapsed_time = t1_2 - t1_1
    
    write_execution_log(folder, file_name, rows_total, file_divisions, cpu_cores, upper_temp_limit_in_kelvin, rows_per_division, elapsed_time, version_info, cpu_logs)

