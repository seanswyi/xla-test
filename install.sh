#!/bin/bash

# Check if the Poetry virtual environment is activated
if [[ -z "$VIRTUAL_ENV" || $(basename "$VIRTUAL_ENV") != "$(basename $(poetry env info --path))" ]]; then
	echo "Activating Poetry shell..."
	        # Starting a new sub-shell with poetry is not ideal for scripting, so we use `poetry run`
	else
		echo "Poetry shell is active."
fi

# Run poetry install within the virtual environment
echo "Running poetry install..."
poetry run poetry install

# Install torch_xla with pip within the Poetry-managed environment
echo "Installing torch_xla via pip..."
poetry run pip install "torch_xla[tpu]~=2.2.0" -f https://storage.googleapis.com/libtpu-releases/index.html

echo "All installations completed."

