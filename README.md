# music-ai
This repository will store project for Music related AI processing

# While running code via python
Create virtual env first
python -m venv .venv-rnn

Activate virtual env
.venv-rnn\Scripts\activate

python.exe -m pip install --upgrade pip

For Basic RNN training program - I have added following modules inside my own created Virtual env
---------------
I have faced issue where python 3.14 is NOT supported by tensorflow. So check what all Pythons are installed on ur machine -
py -0p
--Create venv for 3.11 which is compaible with tensorflow
py -3.11 -m venv music_env_3.11
music_env_3.11\Scripts\activate
python -m pip install tensorflow numpy pandas pretty_midi pyfluidsynth matplotlib seaborn

------------------------------------
Seeing a lot of noisy files, hence updated .gitignore to skip those
git rm -r --cached music_env_3.11
git rm -r --cached data
git add .gitignore
git commit -m "Ignore virtualenv, data and common Python artifacts"