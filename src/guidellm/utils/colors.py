__all__ = ["Colors"]


class Colors:
    INFO: str = "light_steel_blue"
    PROGRESS: str = "dark_slate_gray1"
    SUCCESS: str = "chartreuse1"
    ERROR: str = "orange_red1"


import gzip
from pathlib import Path


def compress_to_gz(input_path: str, encoding: str = "utf-8") -> str:
    input_file = Path(input_path)
    output_file = input_file.with_suffix(input_file.suffix + ".gz")

    with open(input_file, encoding=encoding) as f_in:
        with gzip.open(output_file, "wt", encoding=encoding) as f_out:
            f_out.writelines(f_in)

    print(f"Compressed file saved to: {output_file}")
    return str(output_file)
