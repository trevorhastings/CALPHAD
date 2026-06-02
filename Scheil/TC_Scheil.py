import numpy as np, pandas as pd
from tc_python import TCPython
from tc_python import CompositionUnit, ScheilOptions, ScheilQuantity
import concurrent.futures
import os
import sys
import atexit
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import functools
import time
from typing import Protocol
import math

from thermocalc_library import check_division, check_temperature_limit, check_single_system
from thermocalc_library import write_cpu_log, collect_cpu_logs, write_execution_log
from thermocalc_library import collect_version_info
from thermocalc_library import divide_up_cores
from thermocalc_library import create_parallelization_list, prepare_inputs_with_arbitrary_size
from thermocalc_library import concatenate_divided_data

def cleanup():
    sys.stdout = sys.__stdout__
    log_file.close() # The close method includes a flush method internally

class PropertyResultLike(Protocol):
    '''TC Python internals may change without warning.
    If the instance of the TC class has any of the below methods, then that assigned object
    will pass static type checkers.'''
    def get_value_of(self, name: str) -> float: ...

def Scheil(
        database: str,
        T_upper_limit_K: int,
        T_step: float,
        folder: str,
        elements: list[str],
        model_input) -> pd.DataFrame:
    t0 = time.monotonic()
    with TCPython() as session:
        # ---- TC Property module
        eq_calculation: PropertyResultLike = (session.select_database_and_elements(
            database, elements)
                          .get_system()
                          .with_property_model_calculation("Liquidus and Solidus Temperature")
                          .set_argument('upperTemperatureLimit', T_upper_limit_K)
                          .set_argument('maxNumberOfIterations', 10)
                          .set_composition_unit(CompositionUnit.MOLE_FRACTION))
        cpu_block: list[int] = model_input["Material"].unique()
        col_1: str = 'Liquidus_temperature_(C)'
        col_2: str = 'Solidus_temperature_(C)'
        
        for material in cpu_block:
            comp: np.ndarray = np.array(model_input.loc[model_input['Material'] == material][elements].iloc[0])
            for j in range(len(elements) - 1):
                eq_calculation: PropertyResultLike = eq_calculation.set_composition(
                    elements[j], comp[j])
            try:
                result: PropertyResultLike = eq_calculation.calculate()
                T_liquidus_K = result.get_value_of('Liquidus temperature')
                T_solidus_K = result.get_value_of('Solidus temperature')
                model_input.loc[model_input['Material'] == material, col_1] = T_liquidus_K - 273.15
                model_input.loc[model_input['Material'] == material, col_2] = T_solidus_K - 273.15
            except:
                T_liquidus_K = 4000.0
                T_solidus_K = np.nan
            temperature_starting: float = (np.ceil(T_liquidus_K) - 273.15) + 273.15
            # ---- TC Scheil module
            sch_calculation = (session.select_database_and_elements(database, elements)
                              .get_system()
                              .with_scheil_calculation()
                              .set_start_temperature(temperature_starting)
                              .with_options(ScheilOptions().set_temperature_step(T_step))
                              .set_composition_unit(CompositionUnit.MOLE_FRACTION))
            for j in range(len(elements) - 1):
                sch_calculation: PropertyResultLike  = sch_calculation.set_composition(
                    elements[j], comp[j])
            try:
                result: PropertyResultLike = sch_calculation.calculate()
                result_data = result.get_values_grouped_by_stable_phases_of(
                    ScheilQuantity.temperature(),
                    ScheilQuantity.volume_fraction_of_all_liquid())
                material_mask = model_input['Material'] == material
                material_indices = model_input[material_mask].index
                offset: int = 0
                for phase, values in result_data.items():
                    temps = values.get_x()
                    fractions = values.get_y()
                    valid_pairs = [(t, f) for t, f in zip(temps, fractions)
                                   if not (math.isnan(t) or math.isnan(f))]
                    temps, fractions = zip(*valid_pairs) if valid_pairs else ([], [])
                    temps = [int((t - 273.15) * 100 + 0.5) / 100 for t in temps]
                    fractions = [1 - f for f in fractions]

                    rows_n = len(temps)
                    idxs = material_indices[offset:offset + rows_n]
                    
                    model_input.loc[idxs, 'Temp_(C)'] = list(temps)
                    model_input.loc[idxs, 'Vol_Fra_SOL'] = list(fractions)                    
                    model_input.loc[idxs, 'Label'] = phase
                    offset += rows_n
    
            except Exception as e:
                print(f'Error in calculation for index {material}: {e}')
            finally:
                path_output: str = f'{folder}/SCHEIL_OUT_{max(cpu_block)}.csv'
                model_input.to_csv(path_output, index=False)
        # ---- End CPU loop through list
        print(f'Completed: SCHEIL_OUT_{max(cpu_block)}.csv')
    # ---- End TCPython Session
    elapsed_time_model: float = time.monotonic() - t0
    write_cpu_log(folder, model_input, elapsed_time_model, cpu_block)
    return None

if __name__ == '__main__':

    t1_1: float = time.monotonic()    
    # ---- User inputted parameters
    file_name: str = 'scheil_test_4'
    file_divisions: int = 4
    cpu_cores: int = 4
    version: str = '2024b'
    database = 'TCHEA7' # 'TCHEA7' 'TCAL9'
    T_upper_limit_K: int = 3700
    T_step: float = 1.0
    folder = f'{file_name}_TCscheil_1_0_3'

    if not os.path.exists(folder):
        os.mkdir(folder)
    # ---- Buffer & stream redirection
    log_file = open(f'{folder}/Log_thermocalc.txt', 'w')
    sys.stdout = log_file
    atexit.register(cleanup)
    
    data_input: pd.DataFrame = pd.read_csv(f"{file_name}.csv")
    # ---- Perform checks
    check_division: bool = check_division(file_divisions, cpu_cores)
    check_single_system: bool = check_single_system(data_input)
    check_temperature_limit(data_input, T_upper_limit_K)
    make_database_inquiry: bool = True
    version_info: list[str]
    check_database: bool
    version_info, check_database = collect_version_info(version, database, make_database_inquiry)
    with open(os.path.join(folder, "all_checks_passed.txt"), "w", newline="") as f:
        pass
    # ---- Set up divisions
    elements: list[str] = [col for col in data_input.columns if len(col) <= 2 and data_input.iloc[0][col] > 0]
    rows_per_division, rows_total = divide_up_cores(data_input, file_divisions)
    model_inputs = create_parallelization_list(data_input, rows_per_division)
    model_inputs = prepare_inputs_with_arbitrary_size(model_inputs, 10000)
    partial_property = functools.partial(
        Scheil,
        database,
        T_upper_limit_K,
        T_step,
        folder,
        elements)
    
    log_file.write('\n.\n.\nBegin utilizing multiple CPUs:\n.\n.\n')
    log_file.flush()
    # ---- Worker processes
    with concurrent.futures.ProcessPoolExecutor(cpu_cores) as executor:
        for _ in executor.map(partial_property, model_inputs):
            pass
    # ---- Folder cleanup
    concatenate_divided_data(folder = folder, file_start = 'SCHEIL_OUT_')
    cpu_logs: list[str] = collect_cpu_logs(folder)
    t1_2: float = time.monotonic()
    elapsed_time = t1_2 - t1_1
    print(f'Script execution finished in {elapsed_time:.2f} seconds')
    # ---- Write final log
    write_execution_log(folder, file_name, rows_total, file_divisions, cpu_cores, T_upper_limit_K, rows_per_division, elapsed_time, version_info, cpu_logs)