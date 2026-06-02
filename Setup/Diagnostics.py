# %% Log diagnostic

import sys
import atexit
import traceback
from tc_python import TCPython, ThermodynamicQuantity

def cleanup():
    sys.stdout = sys.__stdout__
    log_file.close()

log_file = open('Log_thermocalc.txt', 'w')
sys.stdout = log_file
atexit.register(cleanup)

try:
    print('Begin TCPython session...')
    with TCPython() as session:
        database = "TCAL9"
        elements = ["Al", "Cu"]
        print('Begin equilbrium...')
        eq_calculation = (session.select_database_and_elements(database, elements)
                         .get_system()
                         .with_single_equilibrium_calculation()
                         .set_condition(ThermodynamicQuantity.temperature(), 298)
                         .set_condition(ThermodynamicQuantity.mole_fraction_of_a_component(
                             elements[0]), 0.40)
                         )
        result = eq_calculation.calculate()
        print(f'Phases: {result.get_phases()}')
except Exception:
    print(traceback.format_exc())

print('Finished log diagnostic')