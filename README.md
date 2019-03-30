# ssdetection-rnn
Automatic detection of sleep spindles using LSTM recurrent neural network.


1. Run script repair\_inta.py to repair INTA marks.
2. Run script generate\_data\_ckpt.py to save fresh checkpoints of both MASS and INTA databases.
3. You can use simple\_train.py as an example for training, or simple\_predict\_from\_ckpt.py as an example for prediction from a checkpoint file.
4. If you have several predictions, you can average them using average\_predictions.py
5. You can check the performance of predictions using the notebook evaluate\_predictions.ipynb

