# Sign-Language (Is for chinese)
A very simple CNN project.

## Note
Simple-OpenCV-Calculator and this project are merged to one. Simple-OpenCV-Calculator will no longer be maintained.

## Outcome
1. Watch Text Demo <a href="https://www.youtube.com/watch?v=xCTUmRXpIvM">here</a>.
2. Watch Calculate Demo <a href="https://www.youtube.com/watch?v=RwmpwDxPSgg">here</a>.

## Requirements
0. Python 3.X
1. Keras
2. OpenCV
3. h5py
4. pyttsx3
5. A good CPU (preferably with a GPU).
6. Patience.... A lot of it.

## Installing the requirements
1. Start your terminal of cmd depending on your os.
  2. If you have a NVidia GPU then make sure you have the prerequisites for Tensorflow GPU installation (Refer to official site). Then use this commmand

    pip install -r requirements_gpu.txt

  3. In case you do not have a GPU then use this command

    pip install -r requirements_cpu.txt

# Use "fun_util.py" to try below mode.
#### Text Mode (Press 't' to go to text mode)
1. In text mode you can create your own words using fingerspellings or use the predefined gestures.
2. The text on screen will be converted to speech on removing your hand from the green box
3. Make sure you keep the same gesture on the green box for 15 frames or else the gesture will not be converted to text.

#### Calculator Mode (Press 'c' to go to calculator mode)
1. To confirm a digit make sure you keep the same gesture for 20 frames. On successful confirmation, the number will appear in the vertical center of the black part of the window.
2. To confirm a number make the "等於" gesture and keep in the green box for 25 frames. You will get used to the timing :P.
3. You can have any number of digits for both first number and second number.
4. Currently there are 4 operators.
5. During operator selection, 1 means '+', 2 means '-', 3 means '*', 4 means '/'.
