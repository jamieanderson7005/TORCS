To run the machine learning there are 3 modes: optimise, continuous and drive.
to run each mode firstly open wtorcs.
Once wtorcs is running and you have selected the race with server1 as the driver:

    run either of these commands for the modes in the console:

    python "directory of torcs_continuous" --optimise
    python "directory of torcs_continuous" --continuous
    python "directory of torcs_continuous" --drive

optimise

optimise is a training loop that has an end which can be changed.
it will run multiple laps of which is specified and once those have completed it will take the best and move onto generation 2 in which the best from generation 1 is used and improved upon and repeats in generations until an end term.

continuous

continuous will run the exact same as optimise however it doesnt have an end condition and will run generations indefinetly.

drive

drive will deploy the best trained model for demo or racing.