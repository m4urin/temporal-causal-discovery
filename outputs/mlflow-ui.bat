call %UserProfile%\Anaconda3\Scripts\activate.bat
call activate causal-discovery
start mlflow ui
timeout /t 1
start http://127.0.0.1:5000
