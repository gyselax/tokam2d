# main.py
# Thin backward-compatible CLI shim. The implementation now lives in the
# installable package (`tokam2d.cli`); install with `pip install -e .` and run
# either `python main.py -i input.yaml -o out/` or the `tokam2d` console script.
from tokam2d.cli import main

if __name__ == "__main__":
    print(
        """
______________________________________________________________
|  _____           _                          ____    ____    |
| |_   _|   ___   | | __   __ _   _ __ ___   |___ \\  |  _ \\   |
|   | |    / _ \\  | |/ /  / _` | | '_ ` _ \\    __) | | | | |  |
|   | |   | (_) | |   <  | (_| | | | | | | |  / __/  | |_| |  |
|   |_|    \\___/  |_|\\_\\  \\__,_| |_| |_| |_| |_____| |____/   |
______________________________________________________________
    """)
    main()
