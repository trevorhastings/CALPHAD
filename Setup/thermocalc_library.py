# -*- coding: utf-8 -*-
"""
Created on Tue May 26 08:50:55 2026

@author: Timet
"""

import os
import pandas as pd

def check_division(file_divisions: int, cpu_cores: int) -> bool:
    if cpu_cores > file_divisions:
        raise ValueError(
            "You have assigned more cores than file divisions. What do you expect the extra cores to do?"
            "Generally the number of cores ought to equal the number of divisions"
            "\n(e.g. 300,000 compositions divided by 48)."
            "\nYou can divide the list into smaller sizes, and however many cores you've assigned will file into the remaining divisions as any of them become available, presumably for memory allocation reasons. Asking for more cores than the number you intend to use in parallel is ridiculous.")
    return True

def check_temperature_limit(data_input, upper_temp_limit_in_kelvin) -> None:
    els: list[str] = [col for col in data_input.columns if len(col) <= 2]
    unary_melting_K = {
        'Be': 1560,
        'B': 2348,
        'C': 3915,
        'Si': 1687,
        'Sc': 1814,
        'Ti': 1941,
        'V': 2183,
        'Cr': 2180,
        'Mn': 1519,
        'Fe': 1811,
        'Co': 1768,
        'Ni': 1728,
        'Cu': 1358,
        'Ge': 1211,
        'As': 1090,
        'Sr': 1050,
        'Y': 1799,
        'Zr': 2128,
        'Nb': 2750,
        'Mo': 2896,
        'Tc': 2430,
        'Ru': 2607,
        'Rh': 2237,
        'Pd': 1828,
        'Ag': 1235,
        'Ba': 1000,
        'Hf': 2506,
        'Ta': 3290,
        'W': 3695,
        'Re': 3459,
        'Os': 3306,
        'Ir': 2739,
        'Pt': 2041,
        'Au': 1337
        }
    dictionary_subset = {el: unary_melting_K[el] for el in els if el in unary_melting_K}
    melting_points = [dictionary_subset[el] for el in dictionary_subset]
    if len(melting_points) > 0:
        if False in [upper_temp_limit_in_kelvin > point for point in melting_points]:
            raise ValueError(
                f'Your selected temperature limit\n{upper_temp_limit_in_kelvin} (K) '
                f'is lower than the melting point (K) of at least one of your unary elements:\n'
                + '\n'.join(f'* {k}: {v}' for k, v in dictionary_subset.items()))
    return None

def check_single_system(data_input: pd.DataFrame) -> bool:
    els: list[str] = [col for col in data_input.columns if len(col) <= 2]
    nonzero_sets = data_input[els].apply(lambda row: frozenset(el for el in els if row[el] > 0), axis=1)
    first_set = nonzero_sets.iloc[0]
    if not all(s == first_set for s in nonzero_sets):
        raise ValueError("Your file includes compositions of differing elements. This script only supports composition lists of exactly the same system. Sending multiple systems of differing active elements to a Thermo-Calc licensing server is likely to cause issues with hanging CPUs, resulting in 1000's of failed evaluations, as it has to restart the system definer with every active element change. If you need multiple systems (perhaps the combinations of ternaries out of an 8-element set, with low resolution), then you should be using another script that intentionally makes this check.")
    return True


def write_cpu_log(folder: str, model_input: pd.DataFrame, elapsed_time_model: float, cpu_block: list[int]) -> None:
    model_path = os.path.join(folder, f"cpu_log_{max(cpu_block)}.txt")
    with open(model_path, "w") as model_log:
        model_log.write(f"Division starting at material {min(cpu_block)}\n")
        model_log.write(f"Division ending at material {max(cpu_block)}\n")
        model_log.write(f"Number of materials evaluated: {len(cpu_block)}\n")
        model_log.write(f"Division execution time: {elapsed_time_model:.2f} seconds\n")
    return None

def collect_cpu_logs(folder: str) -> list:
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

def write_execution_log(folder: str, file_name: str, rows_total: int, file_divisions: int, cpu_cores: int, T_upper_limit_K: int, rows_per_division: int, elapsed_time: float, version_info: list[str], cpu_logs: list[str]) -> None:
    log_path: str = os.path.join(folder, "execution_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write("="*40 + "\n")
        log_file.write("Thermo-Calc TC-Python Execution Log\n")
        log_file.write("="*40 + "\n")
        log_file.write(f"Input file: {file_name}.csv\n")
        log_file.write(f"Number of compositions surveyed: {rows_total}\n")
        log_file.write(f"Number of file divisions: {file_divisions}\n")
        log_file.write(f"CPU cores used: {cpu_cores}\n")
        log_file.write(f"Upper temperature limit (K): {T_upper_limit_K}\n")
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

def collect_version_info(version: str, database: str, make_database_inquiry: bool) -> tuple[list, bool]:
    import tc_python
    def check_database(database: str, databases: list[str]) -> None:
        if not database in databases:
            filtered = sorted([d for d in databases if 'DEMO' not in d and not d.startswith('MO')])
            mo_dbs = sorted([d for d in databases if d.startswith('MO') and 'DEMO' not in d])
            raise ValueError(
                f'The database you selected, "{database}" is not available. Perhaps you have a typo?'
                f'If this surprises you, abort trying run this script and run diagonistics.\n\n'
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
    # Unfortunately, you cannot use "from tc_python import SetUp" and "SetUp().get_databases()" for a database check. SetUp must be used within an active TCPython class, even if you aren't doing any calculations (the documentation isn't explicit about this). It depends on an initialized Java Virtual Machine (JVM) after the gateway is established. "tc_python.__version__", retrieves a static attribute from the tc_python module based on the installation. So there's an extra session call outisde the parallelization in this script. This adds some runtime (e.g. 7 seconds), all because they couldn't be bothered to write a separate class for licensing and database inquiries. This--vexingly--writes additional junk to the console that you won't use since you only want to check that the database information is correct without material calculations. But it's nice to catch this before you start connecting to multiple CPUs.
    if make_database_inquiry:
        with tc_python.TCPython() as session:
            databases = session.get_databases()
            check_database = check_database(database, databases)
            version_lines.append(f"Available databases: {databases}")
    version_lines.append("=" * 40)
    return version_lines, check_database





def divide_up_cores(data_input: pd.DataFrame, file_divisions: int) -> tuple[int, int]:
    rows_total = len(data_input)
    rows_per_division = (rows_total + file_divisions - 1) // file_divisions
    return rows_per_division, rows_total

def create_parallelization_list(data_input: pd.DataFrame, rows: int) -> list[pd.DataFrame]:
    model_inputs: list[pd.DataFrame] = [data_input.iloc[start:(start + rows)].copy() for start in range(0, len(data_input), rows)]
    return model_inputs

def prepare_inputs_with_arbitrary_size(data_parallel: list[pd.DataFrame], multiple: int):
    expanded_data: list = []
    for df in data_parallel:
        expanded_i: pd.Dataframe = df.loc[df.index.repeat(multiple)].reset_index(drop = True)
        expanded_data.append(expanded_i)
    return expanded_data




def concatenate_divided_data(folder: str, file_start: str) -> None:
    file_paths: list[str] = [os.path.join(folder, f) for f in os.listdir(folder) if f.startswith(file_start)]
    data_divided: list[pd.DataFrame] = [pd.read_csv(f) for f in file_paths]
    data_all = pd.concat(data_divided, ignore_index = True)
    if 'Temp_(C)' in data_all.columns:
        data_all.dropna(subset = ['Temp_(C)'], inplace = True)
        data_all = data_all.sort_values(by = ['Material', 'Temp_(C)'])
    data_all.to_csv(os.path.join(folder, "All_data.csv"), index = False)
    return None



if "__name__" == "__main__":
    pass