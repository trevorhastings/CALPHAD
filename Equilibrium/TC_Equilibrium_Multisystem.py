import numpy as np, pandas as pd
from tc_python import *
import concurrent.futures
import os
import functools
import time
from typing import Protocol

# class PropertyResultLike(Protocol):
#     '''For strict type checkers: if the internal TC function type changes, '''
#     def get_value_of(self, name: str) -> float: ...

def Equilibrium(
        database: str,
        phases: list[str],
        temperatures_in_celsius: list[int],
        folder: str,
        elements: list[str],
        model_input) -> pd.DataFrame:
    t0 = time.monotonic()
    current_elements_set = None
    eq_calculation = None
    with TCPython() as session:
        # eq_calculation = (session.select_database_and_elements(database, elements)
        #                   .get_system()
        #                   .with_single_equilibrium_calculation()
        #                   .set_condition(ThermodynamicQuantity.temperature(), 298)
        #                   )
        for i in model_input.index:
            active_elements = sorted([el for el in elements if model_input.loc[i, el] > 0])
            
            if active_elements != current_elements_set:
                try:
                    eq_calculation = (session.select_database_and_elements(database, active_elements)
                                      .get_system()
                                      .with_single_equilibrium_calculation()
                                      .set_condition(ThermodynamicQuantity.temperature(), 298)
                                      )
                    current_elements_set = active_elements
                except Exception as e:
                    print(f"Failed to initialize system for elements {active_elements}: {e}")
                    continue
                
            # comp: np.ndarray = np.array(model_input.loc[i][elements])
            comp: np.ndarray = np.array(model_input.loc[i][active_elements])
            # for j in range(len(active_elements) - 1):
            #     eq_calculation.set_condition(ThermodynamicQuantity
            #                                  .mole_fraction_of_a_component(active_elements[j]), comp[j])
            for j in range(len(active_elements) - 1):
                eq_calculation.set_condition(ThermodynamicQuantity
                                             .mole_fraction_of_a_component(active_elements[j]), comp[j])  
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
                        if any(phase.startswith(p) for p in phases):
                            value = result.get_value_of(
                                ThermodynamicQuantity.volume_fraction_of_a_phase(phase))
                            if phase not in collected_values:
                                collected_values[phase] = {}
                            collected_values[phase][temp] = value
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
        print(f'Completed: Equil_OUT_{model_input.index[0]}.csv')
    elapsed_time_model: float = time.monotonic() - t0
    write_cpu_log(folder, model_input, elapsed_time_model)
    return None

def divide_up_cores(data_input: pd.DataFrame, file_divisions: int) -> tuple[int, int]:
    rows_total = len(data_input)
    rows_per_division = (rows_total + file_divisions - 1) // file_divisions
    return rows_per_division, rows_total

def create_parallelization_list(data_input: pd.DataFrame, rows: int) -> list[pd.DataFrame]:
    model_inputs: list[pd.DataFrame] = [data_input.iloc[start:(start + rows)].copy() for start in range(0, len(data_input), rows)]
    return model_inputs

def write_cpu_log(folder: str, model_input: pd.DataFrame, elapsed_time_model: float) -> None:
    model_path = os.path.join(folder, f"cpu_log_{model_input.index[0]}.txt")
    with open(model_path, "w") as model_log:
        model_log.write(f"Division starting at index {model_input.index[0]}\n")
        model_log.write(f"Number of rows evaluated: {model_input.shape[0]}\n")
        model_log.write(f"Division execution time: {elapsed_time_model:.2f} seconds\n")
    return None

def check_division(file_divisions: int, cpu_cores: int) -> bool:
    if cpu_cores > file_divisions:
        raise ValueError("You have assigned more cores than file divisions. What do you expect the extra cores to do? Generally the number of cores ought to equal the number of divisions (e.g. 300,000 compositions divided by 48). You can divide the list into smaller sizes, and however many cores you've assigned will file into the remaining divisions as any of them become available, presumably for memory allocation reasons. Asking for more cores than the number you intend to use in parallel is ridiculous.")
    return True

def collect_version_info(version: str, database: str, make_database_inquiry: bool) -> tuple[list, bool]:
    import tc_python
    def check_database(database: str, databases: list[str]) -> None:
        if not database in databases:
            filtered = sorted([d for d in databases if 'DEMO' not in d and not d.startswith('MO')])
            mo_dbs = sorted([d for d in databases if d.startswith('MO') and 'DEMO' not in d])
            raise ValueError(
                f'The database you selected, "{database}" is not available. Perhaps you have a typo? If this surprises you, abort trying run this script and run diagonistics.\n\n'
                f'Chemical databases:\n'
                f'{", ".join(filtered)}\n\n'
                f'Mobility databases:\n'
                f'{", ".join(mo_dbs)}'
            )
        return True
    version_lines = []
    version_lines.append("Thermo-Calc Environment Info")
    version_lines.append("=" * 40)
    version_lines.append(f"Declared Thermo-Calc version: {version}")
    try:
        version_lines.append(f"TC-Python reported version: {tc_python.__version__}")
    except Exception as e:
        version_lines.append(f"Error getting TC-Python version: {e}")
    version_lines.append(f"Used database: {database}")\
    # Unfortunately, you cannot use "from tc_python import SetUp" and "SetUp().get_databases()" for a database check. SetUp must be used within an active TCPython class, even if you aren't doing any calculations (the documentation isn't explicit about this). It depends on an initialized Java Virtual Machine (JVM) after the gateway is established. "tc_python.__version__", retrieves a static attribute from the tc_python module based on the installation. So there's an extra session call outisde the parllelization in this script. This adds some runtime (e.g. 7 seconds), but it's nice to catch this before you start connecting to multiple CPUs.
    if make_database_inquiry:
        with tc_python.TCPython() as session:
            databases = session.get_databases()
            check_database = check_database(database, databases)
            version_lines.append(f"Available databases: {databases}")
    version_lines.append("=" * 40)
    return version_lines, check_database

def collect_cpu_logs(version_string: str) -> list:
    import glob
    cpu_logs = []
    try:
        cpu_log_paths = sorted(glob.glob(os.path.join(folder, "cpu_log_*.txt")))
        for path in cpu_log_paths:
            with open(path, "r") as f:
                cpu_logs.extend(f.readlines())
    except Exception as e:
        cpu_logs.append(f"Failed to import chunk logs: {e}\n")
    return cpu_logs

def write_execution_log(folder: str, file_name: str, rows_total: int, file_divisions: int, cpu_cores: int, phases: list[str], rows_per_division: int, elapsed_time: float) -> None:
    log_path: str = os.path.join(folder, "execution_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("="*40 + "\n")
        log_file.write("Thermo-Calc TC-Python Execution Log\n")
        log_file.write("="*40 + "\n")
        log_file.write(f"Input file: {file_name}.csv\n")
        log_file.write(f"Number of compositions surveyed: {rows_total}\n")
        log_file.write(f"Number of file divisions: {file_divisions}\n")
        log_file.write(f"CPU cores used: {cpu_cores}\n")
        log_file.write(f"Phases: {phases}\n")
        log_file.write(f"Compositions per division: {rows_per_division}\n")
        log_file.write(f"Output folder: {folder}\n")
        log_file.write(f"Total execution time: {elapsed_time:.2f} seconds\n")
        log_file.write("="*40 + "\n")
        
        for line in version_info:
            log_file.write(line + "\n")
        
        log_file.write("Individual Core Times\n")
        log_file.write("="*40 + "\n")
        for line in cpu_logs:
            log_file.write(line)
        log_file.write("="*40 + "\n")
    print(f"Log file written to: {log_path}")
    return None

def concatenate_divided_data(folder: str, file_start: str) -> None:
    file_paths: list[str] = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(file_start)]
    data_divided: list[pd.DataFrame] = [pd.read_csv(f) for f in file_paths]
    data_all = pd.concat(data_divided, ignore_index=True)
    data_all.to_csv(os.path.join(folder, "All_data.csv"), index = False)
    return None

if __name__ == '__main__':
    
    t1_1: float = time.monotonic()
    
    # ---- User inputted parameters
    file_name: str = 'Karaman_Dual_Phase_Project_C1C3C4'
    # file_name: str = 'mv_refractorytest_1000'
    # file_name: str = 'dual_phase_panina'
    # file_name: str = 'space_for_singlephase_every20pct_hcp'
    # file_name: str = 'fcc_phase_data'
    # file_name: str = 'quick_fcc_gradient'
    file_divisions: int = 10
    cpu_cores: int = 10
    version: str = '2024b'
    database = 'TCHEA7' # TCHEA7, TCTI6
    phases: list[str] = ['BCC'] # FCC, BCC, HCP, ALTI3, LAVES, CU4TI, GAMMA, LIQUID, SIGMA    
    temperatures_in_celsius: list[int] = [1200]
    # temperatures_in_celsius: list[int] = [300, 400]
    # temperatures_in_celsius: range = range(300, 600 + 100, 100)
    folder = f'{file_name}_TCequi1'
    
    temperatures_in_celsius = sorted(temperatures_in_celsius)
    if not os.path.exists(folder):
        os.mkdir(folder)    
    data_input: pd.DataFrame = pd.read_csv(f"{file_name}.csv")
    
    check_division: bool = check_division(file_divisions, cpu_cores)
    make_database_inquiry: bool = True # Only set this to False if you are running the script *locally*
    version_info: list[str]
    check_database: bool
    version_info, check_database = collect_version_info(version, database, make_database_inquiry)
    with open(os.path.join(folder, "all_checks_passed.txt"), "w", newline="") as f:
        pass
    
    # elements: list[str] = [col for col in data_input.columns if len(col) <= 2 and data_input.iloc[0][col] > 0]
    elements: list[str] = [col for col in data_input.columns if len(col) <= 2]
    
    data_input['system_sig'] = data_input[
        elements].gt(0).apply(
            lambda x: '-'.join(sorted(x.index[x].tolist())), axis=1)
    data_input = data_input.sort_values('system_sig').reset_index(drop=True)
    
    rows_per_division, rows_total = divide_up_cores(data_input, file_divisions)
    model_inputs = create_parallelization_list(data_input, rows_per_division)
    partial_property = functools.partial(
        Equilibrium,
        database,
        phases,
        temperatures_in_celsius,
        folder,
        elements)
    
    with concurrent.futures.ProcessPoolExecutor(cpu_cores) as executor:
        for _ in executor.map(partial_property, model_inputs):
            pass
    
    concatenate_divided_data(folder = folder, file_start = 'EQUIL_OUT_')
    
    cpu_logs: list[str] = collect_cpu_logs(folder)
    
    t1_2: float = time.monotonic()
    elapsed_time = t1_2 - t1_1
    print(f'Script execution finished in {elapsed_time:.2f} seconds')
    
    write_execution_log(folder, file_name, rows_total, file_divisions, cpu_cores, phases, rows_per_division, elapsed_time)


# %%

import os
import pandas as pd
file_paths: list[str] = [os.path.join('Karaman_Dual_Phase_Project_C1C3C4_TCequi1', f) for f in os.listdir('Karaman_Dual_Phase_Project_C1C3C4_TCequi1') if f.startswith('EQUIL_OUT_')]

data_divided: list[pd.DataFrame] = [pd.read_csv(f) for f in file_paths]
data_all = pd.concat(data_divided, ignore_index=True)

data_all = data_all.sort_values(by = ['Alloy','Phase'], ascending = [True, False])




# print('\n'.join(f"{i+1}. {col}" for i, col in enumerate(data_all.columns)))
data_all.to_csv(os.path.join('Karaman_Dual_Phase_Project_C1C3C4_TCequi1', "All_dataA.csv"), index = False)