

@REM stochastic mode, constant LR=1, or decay from 30 down to 1?

@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 1    -e 5 -k 100 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_16tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 32 -n 32 -p 1 -x 3 -r 1    -e 5 -k 100 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_32tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 64 -n 64 -p 1 -x 3 -r 1    -e 5 -k 100 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_64tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1  -e 5 -k 100 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_128tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui

@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 30.0 -a 1.0  -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_16tc_AA  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 1.0 -a 0.001 -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_16tc_AA  -w 10.0 .18 .33 .678 .412  -m batch32_64_16tc_AA\barn_model_64l_128c.pkl --nogui

@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 30.0 -a 1.0  -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames f  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1.0 -a 0.001 -e 5 -k 50 --lines_params 3 --gradient_sharpness 2.0 --save_frames batch32_64_128tc_AA  -w 10.0 .18 .33 .678 .412  -m batch32_64_128tc_AA\barn_model_64l_128c.pkl --nogui



@REM @REM @REM batch, constant LR=1, or decay from 30 down to 1?
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 1    -e 20 -k 200 --lines_params 3  -z 65536   --gradient_sharpness 2.0 --save_frames batch64k_64_16tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 32 -n 32 -p 1 -x 3 -r 1    -e 20 -k 200 --lines_params 3  -z 65536   --gradient_sharpness 2.0 --save_frames batch64k_64_32tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 64 -n 64 -p 1 -x 3 -r 1    -e 20 -k 200 --lines_params 3  -z 65536   --gradient_sharpness 2.0 --save_frames batch64k_64_64tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1  -e 20 -k 200 --lines_params 3  -z 65536   --gradient_sharpness 2.0 --save_frames batch64k_64_128tc_LR1  -w 10.0 .18 .33 .678 .412  --nogui


@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 30.0 -a 1.0  -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_16c_AA  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 16 -n 16 -p 1 -x 3 -r 1.0 -a 0.001 -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_16c_AA  -w 10.0 .18 .33 .678 .412  -m batch64k_64_16c_AA\barn_model_64l_16t_16c.pkl --nogui


@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 30.0 -a 1.0  -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_128c_AA  -w 10.0 .18 .33 .678 .412  --nogui
@REM python image_learn.py -i .\input\barn.png -l 64 -c 0 -t 128 -n 128 -p 1 -x 3 -r 1.0 -a 0.001 -e 20 -k 100 --lines_params 3  -z 65536   --gradient_sharpness 2.0   --save_frames batch64k_64_128c_AA  -w 10.0 .18 .33 .678 .412  -m batch64k_64_128c_AA\barn_model_64l_128t_128c.pkl --nogui


