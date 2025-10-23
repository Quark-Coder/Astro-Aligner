call astro_env\Scripts\activate.bat

pyinstaller --onefile --windowed --icon=icon.ico ^
    --exclude-module matplotlib ^
    --exclude-module scipy ^
    --exclude-module pytest ^
    --exclude-module ipykernel ^
	--add-data "icon.ico;." ^
    --exclude-module jupyter ^
    --upx-dir="E:\Programs\upx" ^
    AstroAligner_v1.0.py
	
pause