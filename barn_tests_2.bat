@REM  COMMENTED OUT LINES ARE FOR REFERENCE AND FROM barn_tests_1.bat, 
@REM  ACTIVE TESTS (below this section) are randomized training interpolation versions of these (subset of those w/128 struct & color units):

@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1  -e 5 -k 100 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_128tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 30.0 -a 1.0  -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_128tc_AA  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1.0 -a 0.001 -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_128tc_AA  -w 10.0 .18 .33 .678 .412  -m batch32_64_128tc_AA\barn_model_64l_128c.pkl --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1  -e 20 -k 200 --lines_params 3  -z 65536   --gradient_sharpness 2.0 --save_frames batch64k_64_128tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 30.0 -a 1.0  -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_128c_AA  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1.0 -a 0.001 -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_128c_AA  -w 10.0 .18 .33 .678 .412  -m batch64k_64_128c_AA\barn_model_64l_128t_128c.pkl --nogui



@REM # TESTING RANDOM IMAGE SAMPLING 

@REM THESE will re-run:  WHEN RANDOM SAMPLING IS IMPLEMENTED
python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r 1  -e 5 -k 100 --lines_params 3              --save_frames batch32_64_128tc_LR1_t200k  -w 10.0 .18 .33 .678 .412  --nogui
python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r 1  -e 20 -k 200 --lines_params 3  -z 65536   --save_frames batch64k_64_128tc_LR1_t200k -w 10.0 .18 .33 .678 .412  --nogui



python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r 30.0 -a 1.0  -e 5 -k 50 --lines_params 3 --save_frames batch32_64_128tc_AA_t200k  -w 10.0 .18 .33 .678 .412  --nogui
python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r 1.0 -a 0.001 -e 5 -k 50 --lines_params 3 --save_frames batch32_64_128tc_AA_t200k  -w 10.0 .18 .33 .678 .412  -m batch32_64_128tc_AA_t200k\barn_model_64l_128t_128c.pkl --nogui


python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r  30.0 -a 1.0  -e 20 -k 100 --lines_params 3  -z 65536     --save_frames batch64k_64_128c_AA_t200k  -w 10.0 .18 .33 .678 .412  --nogui  -m batch64k_64_128c_AA_t200k\barn_model_64l_128t_128c.pkl 
python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -q 196608 -x 3 -r 1.0 -a 0.001 -e 20 -k 100 --lines_params 3  -z 65536     --save_frames batch64k_64_128c_AA_t200k  -w 10.0 .18 .33 .678 .412  -m batch64k_64_128c_AA_t200k\barn_model_64l_128t_128c.pkl --nogui
