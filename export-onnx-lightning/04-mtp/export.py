"""Export the separate BF16 attention+MoE MTP draft stage."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.export import main


if __name__ == "__main__":
    main(variant="mtp")
