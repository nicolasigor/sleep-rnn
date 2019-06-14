#!/bin/bash
screen -S screen_script1 -d -m -- bash -c ' source activate tf ; python script1.py  ; exec $SHELL' ; 
screen -S screen_script2 -d -m -- bash -c ' source activate tf ; python script2.py  ; exec $SHELL' ;
screen -S screen_script3 -d -m -- bash -c ' source activate tf ; python script3.py  ; exec $SHELL' ; 
screen -S screen_script4 -d -m -- bash -c ' source activate tf ; python script4.py  ; exec $SHELL' 