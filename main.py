# %%
# main.py
# import numpy as np
from pathlib import Path

from src.simulation.initialize_fields import FieldInitiator
from src.simulation.run_simulation import SimulationRunner
from src.model.pde import HasegawaWakatani, modifiedHasegawaWakatani, fluxBalancedHasegawaWakatani, SOL
from src.interfaces.input_reader import StaticParams
from src.interfaces.output_saver import OutputSaver

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

from argparse import ArgumentParser

def main():
    start = time.time()

    # 1) Read the input file
    print("\n\033[0;32m --- Reading input file, setting simulation domain... ---\n\033[0m")
    input_path, save_dir = get_input_path_and_output_folder()
    params = StaticParams(input_path)
    params.logger.info(params)

    # 2) Initialize output saver
    print("\033[0;32m --- Setting up output saver... ---\n\033[0m")
    saver = OutputSaver(params, save_dir)

    # 3) Initialize fields
    print("\n\033[0;32m --- Initializing fields... ---\n\033[0m")
    field_initiator = FieldInitiator(params)
    params.logger.info(field_initiator)
    initial_fields = field_initiator.initial_fields()

    # 4) Run the simulation
    print("\n\033[0;32m --- Running simulation... ---\n\033[0m")
    sim_runner = SimulationRunner(params, saver=saver)
    final_fields = sim_runner.run(initial_fields)

    # 5) If the simulation is successful, concatenate the output files
    print("\n\033[0;32m --- Saving output files... ---\n\033[0m")
    saver.save_output_data_file()
    saver.save_metadata_file()
    saver.save_user_input_file()

    print("Time taken: ", time.time()-start)
    print("Simulation complete.")
    print(f'Outputs saved in {saver.output_folder}')

def get_input_path_and_output_folder():
    parser = ArgumentParser(description="Run TOKAM2D")
    parser.add_argument("-i","--input_file",
                        action="store",
                        nargs="?",
                        default=None,
                        type=Path,
                        help="input file (YAML format)")
    parser.add_argument("-o","--output_folder",
                        action="store",
                        nargs="?",
                        default=None,
                        type=Path,
                        help="output folder")
    args = parser.parse_args()

    if args.input_file is None:
        raise ValueError("No input file provided.")

    return args.input_file, args.output_folder

if __name__ == "__main__":

    print(
    """
______________________________________________________________
|  _____           _                          ____    ____    |
| |_   _|   ___   | | __   __ _   _ __ ___   |___ \  |  _ \   |
|   | |    / _ \  | |/ /  / _` | | '_ ` _ \    __) | | | | |  |
|   | |   | (_) | |   <  | (_| | | | | | | |  / __/  | |_| |  |
|   |_|    \___/  |_|\_\  \__,_| |_| |_| |_| |_____| |____/   |
______________________________________________________________
    """)

    main()

# %%
