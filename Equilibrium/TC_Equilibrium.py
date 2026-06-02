import numpy as np
import pandas as pd
from tc_python import TCPython
from tc_python import ThermodynamicQuantity
import concurrent.futures
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
import functools
import time
from typing import Protocol

from thermocalc_library import check_division, check_single_system
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

def Equilibrium(
        database: str,
        phases: list[str],
        temperatures_in_celsius: list[int],
        folder: str,
        elements: list[str],
        model_input) -> pd.DataFrame:
    t0 = time.monotonic()
    with TCPython() as session:
        eq_calculation = (session.select_database_and_elements(database, elements)
                          .get_system()
                          .with_single_equilibrium_calculation()
                          .set_condition(ThermodynamicQuantity.temperature(), 298)
                          )
        cpu_block: list[int] = list(model_input.index)
        for i in model_input.index:
            comp: np.ndarray = np.array(model_input.loc[i][elements])
            for j in range(len(elements) - 1):
                eq_calculation.set_condition(ThermodynamicQuantity
                                             .mole_fraction_of_a_component(elements[j]), comp[j])  
            try:
                collected_values = {}
                # ---- for each temp in the list
                for temp in temperatures_in_celsius:
                    eq_calculation = eq_calculation.set_condition(ThermodynamicQuantity
                                                                  .temperature(), temp + 273.15)
                    # ---- calculate
                    result = eq_calculation.calculate()
                    # ---- volume fractions
                    phase_names = result.get_phases()
                    for phase in phase_names:
                        # ---- for known phases
                        if any(phase.startswith(p) for p in phases):
                            value = result.get_value_of(ThermodynamicQuantity.
                                                        volume_fraction_of_a_phase(phase))
                            if phase not in collected_values:
                                collected_values[phase] = {}
                            collected_values[phase][temp] = value
                        # ---- if you don't know the phase
                        # value = result.get_value_of(ThermodynamicQuantity.
                        #                             volume_fraction_of_a_phase(phase))
                        # if phase not in collected_values:
                        #     collected_values[phase] = {}
                        # collected_values[phase][temp] = value
                    for phase in collected_values:
                        value = collected_values[phase].get(temp, None)
                        model_input.at[i, f'VolFra_{phase}_at_{temp}'] = value
                    # ---- freeze
                    eq_calc_frozen = result.change_temperature(20 + 273.15)
                    # ---- molar volume
                    for phase in phase_names:
                        if any(phase.startswith(p) for p in phases):
                            value = collected_values[phase][temp]
                            if value > 0:
                                mv = eq_calc_frozen.get_value_of(
                                    ThermodynamicQuantity.molar_volume_of_phase(phase))
                                model_input.at[i, f'MolVol_{phase}_from_{temp}_to_25'] = mv
                    # ---- COPAMF & SFOCIP
                    # for phase in phase_names:
                    #     if any(phase.startswith(p) for p in phases):
                    #         value = collected_values[phase][temp]
                    #         if value > 0:
                    #             for el in elements:
                    #                 mf = eq_calc_frozen.get_value_of(ThermodynamicQuantity
                    #                                     .composition_of_phase_as_mole_fraction(
                    #                                         phase, el))
                    #                 model_input.at[i, f'Sch_{phase}_COPAMF_{el}'] = mf
                    #             for sublattice in [1, 2]:
                    #                 for el in elements:
                    #                     sf = eq_calc_frozen.get_value_of(ScheilQuantity
                    #                                         .site_fraction_of_component_in_phase(
                    #                                             phase, el, sublattice))
                    #                     model_input.at[i, f'Sch_{phase}_sub{sublattice}_SFOCIP_{el}'] = sf
            except Exception as e:
                print(f'Error in calculation for index {i}: {e}')
            finally:
                model_input.to_csv(f'{folder}/EQUIL_OUT_{model_input.index[0]}.csv', index = False)
        # ---- End CPU loop through list
        print(f'Completed: Equil_OUT_{model_input.index[0]}.csv')
    # ---- End TCPython Session
    elapsed_time_model: float = time.monotonic() - t0
    write_cpu_log(folder, model_input, elapsed_time_model, cpu_block)
    return None

if __name__ == '__main__':
    
    t1_1: float = time.monotonic()
    
    # ---- User inputted parameters
    file_name: str = 'TiVCo_n3_d20_single_system'
    file_divisions: int = 10
    cpu_cores: int = 10
    version: str = '2024b'
    database = 'TCHEA7'
    phases: list[str] = ['BCC'] # FCC, BCC, HCP, ALTI3, LAVES, CU4TI, GAMMA, LIQUID, SIGMA
    temperatures_in_celsius: list[int] = [1000]
    # temperatures_in_celsius: list[int] = [300, 400]
    # temperatures_in_celsius: range = range(300, 600 + 100, 100)
    folder = f'{file_name}_TCequi1'
    
    temperatures_in_celsius = sorted(temperatures_in_celsius)
    if not os.path.exists(folder):
        os.mkdir(folder)    
    data_input: pd.DataFrame = pd.read_csv(f"{file_name}.csv")
    # ---- Perform checks
    check_division: bool = check_division(file_divisions, cpu_cores)
    check_single_system: bool = check_single_system(data_input)
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
        Equilibrium,
        database,
        phases,
        temperatures_in_celsius,
        folder,
        elements)
    # ---- Worker processes
    with concurrent.futures.ProcessPoolExecutor(cpu_cores) as executor:
        for _ in executor.map(partial_property, model_inputs):
            pass
    # ---- Folder cleanup
    concatenate_divided_data(folder = folder, file_start = 'EQUIL_OUT_')
    cpu_logs: list[str] = collect_cpu_logs(folder)
    t1_2: float = time.monotonic()
    elapsed_time = t1_2 - t1_1
    print(f'Script execution finished in {elapsed_time:.2f} seconds')
    # ---- Write final log
    write_execution_log(folder, file_name, rows_total, file_divisions, cpu_cores, phases, rows_per_division, elapsed_time, version_info, cpu_logs)