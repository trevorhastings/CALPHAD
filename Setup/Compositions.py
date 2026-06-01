# %%

import numpy as np
import pandas as pd

def generate_barycentric_grid_iterative(input_dimension_number: int, resolution: int = 5) -> np.ndarray:
    stack = [([], resolution)]
    results = []

    while stack:
        current_coords, remaining_sum = stack.pop()
        
        if len(current_coords) == input_dimension_number - 1:
            results.append(current_coords + [remaining_sum])
            continue
        
        for i in range(remaining_sum, -1, -1):
            stack.append((current_coords + [i], remaining_sum - i))
    
    return np.array(results) / resolution

def generate_phase_space(elements: list[str] = ['Cr', 'Fe', 'Co', 'Ni'], resolution: int = 5, rounded_decimal: int = None) -> pd.DataFrame:
    bary_grid: np.ndarray = generate_barycentric_grid_iterative(
        input_dimension_number = len(elements), resolution = resolution)
    if rounded_decimal is not None:
        bary_grid = np.round(bary_grid, rounded_decimal)
        errors = 1.0 - bary_grid.sum(axis = 1)
        first_nonzero = np.argmax(bary_grid != 0, axis = 1)
        bary_grid[np.arange(bary_grid.shape[0]), first_nonzero] += errors
    phase_space: pd.Dataframe = pd.DataFrame(bary_grid, columns = elements)
    return phase_space

PERIODIC_ORDER: dict = {
    1:'H',2:'He',
    3:'Li',4:'Be',
    5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',
    11:'Na',12:'Mg',
    13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',
    19:'K',20:'Ca',
    21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',
    31:'Ga',32:'Ge',33:'As',34:'Se',35:'Br',36:'Kr',
    37:'Rb',38:'Sr',
    39:'Y',40:'Zr',41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',
    49:'In',50:'Sn',51:'Sb',52:'Te',53:'I',54:'Xe',
    55:'Cs',56:'Ba',
    57:'La',58:'Ce',59:'Pr',60:'Nd',61:'Pm',62:'Sm',63:'Eu',64:'Gd',65:'Tb',66:'Dy',67:'Ho',68:'Er',69:'Tm',70:'Yb',
    71:'Lu',72:'Hf',73:'Ta',74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',
    81:'Tl',82:'Pd',83:'Bi',84:'Po',85:'At',86:'Rn',
    87:'Fr',88:'Ra',
    89:'Ac',90:'Th',91:'Pa',92:'U',93:'Np',94:'Pu',95:'Am',96:'Cm',97:'Bk',98:'Cf',99:'Es',100:'Fm',101:'Md',102:'No',
    103:'Lr',104:'Rf',105:'Db',106:'Sg',107:'Bh',108:'Hs',109:'Mt',110:'Ds',111:'Rg',112:'Cn',
    113:'Nh',114:'Fl',115:'Mc',116:'Lv',117:'Ts',118:'Og',
    }

# %%

elements = ['Ti','V','Co']
resolution = 200
space = generate_phase_space(elements = elements, resolution = resolution, rounded_decimal = 3)

custom_sort = list(PERIODIC_ORDER.values())
order_map = {el: i for i, el in enumerate(custom_sort)}

space['system_sig'] = space[elements].gt(0).apply(
    lambda row: '-'.join(sorted(
        row.index[row].tolist(), 
        key = lambda x: order_map.get(x, 118)
    )), axis = 1)

space['system_size'] = space[elements].gt(0).sum(axis=1)

space = space.sort_values(by = 'system_sig', key = lambda col: col.map(lambda s: (len(parts := s.split('-')), *[custom_sort.index(e) for e in parts]))).reset_index(drop=True)

sort_columns = elements[:-1]
space = space.groupby('system_sig', sort=False)[space.columns.tolist()].apply(
    lambda group: group.sort_values(by=sort_columns, ascending=True)).reset_index(drop=True)

space.insert(0, 'Material', [x + 1 for x in range(space.shape[0])])
space_all_nonlargest_systems = space[space['system_size'] != max(space['system_size'].values)]
space_single_large_system = space[space['system_size'] == max(space['system_size'].values)]
systems_a = space[space['system_size'] == 1]
systems_b = {value: group.reset_index(drop = True) for value, group in space.groupby('system_sig') if len(value) > 2}

# %%

export_multisystem_all = True
export_nonlarge_systems = True
export_large_system = True
export_all_subsystems = True

if export_multisystem_all:
    space.to_csv(f'{''.join(elements)}_n{len(elements)}_d{resolution}_all_systems.csv', index = False)
if export_nonlarge_systems:
    space_all_nonlargest_systems.to_csv(f'{''.join(elements)}_n{len(elements)}_d{resolution}_nonlargest_systems.csv', index = False)
if export_large_system:
    space_single_large_system.to_csv(f'{''.join(elements)}_n{len(elements)}_d{resolution}_single_system.csv', index = False)
if export_all_subsystems:
    systems_a.to_csv(f'{''.join(elements)}_n{len(elements)}_d{resolution}_system_unaries.csv', index = False)
    for value, subspace in systems_b.items():
        subspace.to_csv(f'{''.join(elements)}_n{len(elements)}_d{resolution}_system_{value}.csv', index = False)