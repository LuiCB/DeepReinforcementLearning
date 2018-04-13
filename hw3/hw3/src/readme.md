### HOW TO RUN THE TRAINING PROCESS

#### imitation.py
    > the learning rate is fixed after our experiment and the final value is 0.001 with Adam optimizer. 
    > to run the model and the training process, just do "python imitation.py"

#### reinforce.py
    > we fix the hyper-parameters like the learning rate, discont rate and the test interval of 500 training epoches.

#### a2c.py
    > we fix the architecture of the critic model and the actor model is adpated from the configuration of the expert model in question 1
    > the command line argument can be passed via the flags.


### VIDEOs
    > the "./video" directory contains the videos of the question 1 and 2
    > ./video/video-Q1 contains video clips of the question 1
    > ./video/video-Q2 contains video clips of the question 2