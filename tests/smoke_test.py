from pathlib import Path

from tokam2d.simulation.initialize_fields import FieldInitiator
from tokam2d.simulation.run_simulation import SimulationRunner
from tokam2d.model.pde import HasegawaWakatani, modifiedHasegawaWakatani, fluxBalancedHasegawaWakatani, SOL
from tokam2d.interfaces.input_reader import StaticParams
from tokam2d.interfaces.output_saver import OutputSaver

import time

from argparse import ArgumentParser

def main():
    start = time.time()

    # Get a list of file in input_test folder
    input_test_folder = Path(__file__).parent / "input_test"
    input_paths = list(input_test_folder.glob("*.yaml"))
    
    for input_path in input_paths:
        print(f"Input file: {input_path.stem}")
        # Read the input file
        params = StaticParams(input_path, quiet=True)

        # Initialize fields
        initial_fields = FieldInitiator(params).initial_fields()

        # Run the simulation
        sim_runner = SimulationRunner(params)
        _ = sim_runner.run(initial_fields)
        print("OK")

if __name__ == "__main__":
    main()