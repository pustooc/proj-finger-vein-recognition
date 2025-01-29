# To setup and run the experiment

1. Pull the project code into your directory

```
git pull https://github.com/pustooc/proj-finger-vein-recognition.git
```

2. Ensure that Python 3.11.1 is installed, either globally or through pyenv

3. Set up your virtual environment

```
python -m venv venv
```

4. Activate your virtual environment and set terminal variables

```
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

5. Install the required packages

```
pip install requirements.txt
```

6. In the project root, create a `data/images` folder, and move the CASIA dataset images into it. Also create a `data/logs` folder

7. Run tests with

```
pytest test/test_index.py
```

8. If all tests correctly pass, run the experiment with

```
python src/index.py
```

9. When ready to shut down the experiment, deactivate the virtual environment

```
deactivate
```
