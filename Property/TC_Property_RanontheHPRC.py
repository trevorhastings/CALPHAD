import numpy as np
import pandas as pd
from tc_python import *
import concurrent.futures
import os
import functools
import time

def Property(
        database: str,
        upper_temp_limit_in_kelvin: int,
        folder: str,
        elements: str,
        model_input) -> pd.DataFrame:
    t0 = time.monotonic()
    with TCPython() as session:
        eq_calculation = None
        
        for i in model_input.index:
            eq_calculation = (session.select_database_and_elements(database, elements)
                        .get_system()
                        .with_property_model_calculation("Liquidus and Solidus Temperature")
                        .set_argument('upperTemperatureLimit', upper_temp_limit_in_kelvin)
                        .set_argument('maxNumberOfIterations', 10)
                        .set_composition_unit(CompositionUnit.MOLE_FRACTION))

            comp: np.ndarray = np.array(model_input.loc[i][elements])
            for j in range(len(elements) - 1):
                eq_calculation = eq_calculation.set_composition(elements[j], comp[j])
            try:
                result = eq_calculation.calculate()
                new_col1 = 'Liquidus_temperature_(K)'
                new_col2 = 'Solidus_temperature_(K)'
                model_input.at[i, new_col1] = result.get_value_of('Liquidus temperature')
                model_input.at[i, new_col2] = result.get_value_of('Solidus temperature')
            except Exception as e:
                print(f'Error in calculation for index {i}: {e}')
            finally:
                model_input.to_csv(f'{folder}/PROP_OUT_{model_input.index[0]}.csv', index = False)
                continue
        print(f'Completed: PROP_OUT_{model_input.index[0]}.csv')
    
    elapsed = time.monotonic() - t0
    log_path = os.path.join(folder, f"cpu_log_{model_input.index[0]}.txt")
    with open(log_path, "w") as chunk_log:
        chunk_log.write(f"Chunk starting at index {model_input.index[0]}\n")
        chunk_log.write(f"Number of rows evaluated: {model_input.shape[0]}\n")
        chunk_log.write(f"Chunk execution time: {elapsed:.2f} seconds\n")
    return model_input

def divide_up_cores(data_input: pd.DataFrame, file_divisions: int) -> tuple[int, int]:
    rows_total = len(data_input)
    rows_per_division = (rows_total + file_divisions - 1) // file_divisions
    return rows_per_division, rows_total

def create_parallelization_list(data_input: pd.DataFrame, rows: int) -> list[pd.DataFrame]:
    data_input: pd.dataFrame = pd.read_csv(f"{file_name}.csv")
    model_inputs: list[pd.DataFrame] = [data_input.iloc[start:(start + rows)].copy() for start in range(0, len(data_input), rows)]
    return model_inputs

def check_single_system(data_input: pd.DataFrame) -> bool:
    els: list[str] = [col for col in data_input.columns if len(col) <= 2]
    nonzero_sets = data_input[els].apply(lambda row: frozenset(el for el in els if row[el] > 0), axis=1)
    first_set = nonzero_sets.iloc[0]
    if not all(s == first_set for s in nonzero_sets):
        raise ValueError('Nonzero active elements differ between rows. Use the dynamic-element version of the script instead.')
    return True

def collect_version_info(version_string: str, database: str) -> list:
    import tc_python
    version_lines = []
    version_lines.append("Thermo-Calc Environment Info")
    version_lines.append("=" * 40)
    version_lines.append(f"Declared Thermo-Calc version: {version_string}")
    try:
        version_lines.append(f"TC-Python reported version: {tc_python.__version__}")
    except Exception as e:
        version_lines.append(f"Error getting TC-Python version: {e}")
    version_lines.append(f"Used database: {database}")
    try:
        with tc_python.TCPython() as session:
            dbs = session.get_databases()
            version_lines.append(f"Available databases: {dbs}")
    except Exception as e:
        version_lines.append(f"Error during database inquiry: {e}")

    version_lines.append("=" * 40)
    return version_lines

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

def write_execution_log(folder: str, file_name: str, rows_total: int, file_divisions: int, cpu_cores: int, upper_temp_limit_in_kelvin: int, rows_per_division: int, elapsed_time: float) -> None:
    log_path: str = os.path.join(folder, "execution_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("="*40 + "\n")
        log_file.write("Thermo-Calc TC-Python Execution Log\n")
        log_file.write("="*40 + "\n")
        log_file.write(f"Input file: {file_name}.csv\n")
        log_file.write(f"Number of compositions surveyed: {rows_total}\n")
        log_file.write(f"Number of file divisions: {file_divisions}\n")
        log_file.write(f"CPU cores used: {cpu_cores}\n")
        log_file.write(f"Upper temperature limit (K): {upper_temp_limit_in_kelvin}\n")
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

if __name__ == '__main__':
    
    t1_1: float = time.monotonic()
    
    '''User-defined variables'''
    file_name: str = 'htmdec_y2_tc_property_test_2rows'
    file_divisions: int = 2
    cpu_cores: int = 2
    version_string = '2024a'
    database = 'TCHEA7'
    upper_temp_limit_in_kelvin: int = 2200
    
    # try:
    #     with tc_python.TCPython() as session:
    #         databases: list[str] = session.get_databases()
    #         assert database in databases
    # except Exception as e:
    #     print(f"Error during database inquiry: {e}")
    
    folder = f'{file_name}_TCprop'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    data_input: pd.DataFrame = pd.read_csv(f"{file_name}.csv")
    check_single_system(data_input)
    elements: list[str] = [col for col in data_input.columns if len(col) <= 2 and data_input.iloc[0][col] > 0]

    rows_per_division, rows_total = divide_up_cores(data_input, file_divisions)
    model_inputs = create_parallelization_list(file_name, rows_per_division)
    partial_property = functools.partial(
        Property,
        database,
        upper_temp_limit_in_kelvin,
        folder,
        elements)
    version_info: list[str] = collect_version_info(version_string, database)
    with concurrent.futures.ProcessPoolExecutor(cpu_cores) as executor:
        results: list[pd.DataFrame] = list(executor.map(partial_property, model_inputs))
    
    results_all: pd.DataFrame = pd.concat(results, ignore_index = True)
    results_all.to_csv(f'{folder}/All_data.csv')
    cpu_logs: list[str] = collect_cpu_logs(folder)
    
    t1_2: float = time.monotonic()
    elapsed_time = t1_2 - t1_1
    print(f'Script execution finished in {elapsed_time:.2f} seconds')
    
    write_execution_log(folder, file_name, rows_total, file_divisions, cpu_cores, upper_temp_limit_in_kelvin, rows_per_division, elapsed_time)
    
