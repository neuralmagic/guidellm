import os
import re
from datetime import datetime
from pathlib import Path

import toml
from loguru import logger


def get_build_type():
    return os.getenv("GUIDELLM_BUILD_TYPE", "dev")


def get_build_number():
    return os.getenv("GUIDELLM_BUILD_NUMBER", "0")


def construct_project_name_and_version(build_type, build_number, current_version):
    if not re.match(r"^\d+\.\d+\.\d+$", current_version):
        raise ValueError(
            f"Version '{current_version}' does not match the "
            f"semantic versioning pattern '#.#.#'",
        )

    if build_type == "dev":
        project_name = "guidellm"
        version = f"{current_version}.dev{build_number}"
    elif build_type == "nightly":
        project_name = "guidellm"
        date_str = datetime.now().strftime("%Y%m%d")
        version = f"{current_version}.a{date_str}"
    elif build_type == "release_candidate":
        project_name = "guidellm"
        date_str = datetime.now().strftime("%Y%m%d")
        version = f"{current_version}.rc{date_str}"
    elif build_type == "release":
        project_name = "guidellm"
        version = current_version
    else:
        raise ValueError(f"Unknown build type: {build_type}")

    return project_name, version


def update_pyproject_toml(project_name, version):
    try:
        with Path("pyproject.toml").open() as file:
            data = toml.load(file)

        data["project"]["name"] = project_name
        data["project"]["version"] = version

        with Path("pyproject.toml").open("w") as file:
            toml.dump(data, file)

        logger.info(
            f"Updated project name to: {project_name} and version to: {version}",
        )
    except (FileNotFoundError, toml.TomlDecodeError) as e:
        logger.error(f"Error reading or writing pyproject.toml: {e}")
        raise


def main():
    build_type = get_build_type()
    build_number = get_build_number()

    with Path("pyproject.toml").open() as file:
        pyproject_data = toml.load(file)

    current_version = pyproject_data["project"]["version"]
    project_name, version = construct_project_name_and_version(
        build_type,
        build_number,
        current_version,
    )

    if build_type != "release":
        update_pyproject_toml(project_name, version)


if __name__ == "__main__":
    main()
