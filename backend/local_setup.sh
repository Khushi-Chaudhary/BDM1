#! /bin/sh
echo "======================================================================"
echo "Welcome to the setup. This will set up the local virtual env."
echo "And then it will install all the required python libraries."
echo "You can rerun this without any issues."
echo "----------------------------------------------------------------------"

# Measure the time before installing Python libraries
start_time=$(date +%s)

if [ -d ".venv" ]; then
    echo ".venv folder exists. Installing using pip"
else
    echo "creating .venv and install using pip"
    python3 -m venv .venv
fi

# Activate virtual env
echo "Activating virtual environment"
. .venv/bin/activate
export ENV=development
# Upgrade the PIP (if you uncomment this)
#echo "Upgrading PIP..."
#pip install --upgrade pip
echo "Installing Python libraries from requirements.txt..."
pip install -r requirements.txt

# Print installed packages
echo "======================================================================"
echo "Installed Python packages:"
pip freeze
echo "======================================================================"

# Measure the time after installation
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Installation completed in $elapsed_time seconds."

# Run the app
echo "Running main.py..."
python3 main.py || { echo "Failed to start main.py"; exit 1; }

# Work done, deactivate virtual env
deactivate