python3 -m venv .

@REM Activate the virtual environment
@REM Path: activate.bat
.\Scripts\activate

@REM Install the required packages
@REM Path: install.bat
pip install -r requirements.txt

@REM Echo that it is done
echo Done installing the required packages

@REM Pause the script
pause
