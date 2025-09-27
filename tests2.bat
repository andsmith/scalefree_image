python .\image_learn.py -i .\input\test_3_lines.png -l 4  -t 10 -n 10 -p 15 -x 8 -e 100  -k 300  -r 1.0 -a .001 -z 8 --gradient_sharpness 1.0 --save_frames test_4_lines_2_param_gs1_anneal --nogui
python .\image_learn.py -i .\input\test_3_lines.png -l 4  -t 10 -n 10 -p 15 -x 8 -e 100  -k 300  -r 1.0 -a .001 -z 8 --gradient_sharpness 2.0 --save_frames test_4_lines_2_param_gs2_anneal --nogui
python .\image_learn.py -i .\input\test_3_lines.png -l 4  -t 10 -n 10 -p 15 -x 8 -e 100  -k 300  -r 1.0 -a .001 -z 8 --gradient_sharpness 5.0 --save_frames test_4_lines_2_param_gs5_anneal --nogui
python .\image_learn.py -i .\input\test_3_lines.png -l 4  -t 10 -n 10 -p 15 -x 8 -e 100  -k 300  -r 1.0 -a .001 -z 8 --gradient_sharpness 10.0 --save_frames test_4_lines_2_param_gs10_anneal --nogui

