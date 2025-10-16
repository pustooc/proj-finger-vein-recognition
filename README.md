# Finger vein authentication with a custom CNN

A CNN is built with a fully custom architecture, then trained to authenticate users via finger veins. The focus is on low compute requirements.

To note:
- Accuracy is the only reported metric for benchmarking, but the full confusion matrix can be used for debugging. Size would be 100 * 2 (for 100 users)
- Further architectural decisions are described in dissertation

# To setup and run the experiment

1. Pull the project code into your desired directory

```
git pull https://github.com/pustooc/proj-finger-vein-recognition.git
```

2. Change directory to the project root

```
cd proj-finger-vein-recognition
```

3. Ensure that Python 3.11.1 is installed, either globally or through pyenv

4. Set up your virtual environment

```
python -m venv venv
```

5. Activate your virtual environment and set terminal variables

```
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

6. Install the required packages

```
pip install requirements.txt
```

7. In the project root, create a `data/images` folder, and move the CASIA dataset images into it. Also create a `data/logs` folder

8. Run tests with

```
pytest test/test_index.py
```

9. If all tests correctly pass, run the experiment with

```
python src/index.py
```

10. When ready to shut down the experiment, deactivate the virtual environment

```
deactivate
```
