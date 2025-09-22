python .\image_learn.py -i .\input\washington.png -t linear -e 2 -d 8  -n 10 --learning_rate .1 -p 4 -x 3 --save_frames movies --cycles 100
python .\image_learn.py -i .\input\washington.png -t linear -e 2 -d 32 -n 10 --learning_rate .1 -p 4 -x 3 --save_frames movies --cycles 100
python .\image_learn.py -i .\input\washington.png -t linear -e 3 -d 128 -n 32 --learning_rate .1 -p 4 -x 3 --save_frames movies --cycles 100
python .\image_learn.py -i .\input\washington.png -t linear -e 3 -d 512 -n 64 --learning_rate .1 -p 4 -x 3 --save_frames movies --cycles 100

ffmpeg -y -framerate 32 -i movies\washington_linear_8d_10h_cycle-%08d.png -c:v libx264 -pix_fmt yuv420p washington_linear_8.mp4
ffmpeg -y -framerate 32 -i movies\washington_linear_32d_10h_cycle-%08d.png -c:v libx264 -pix_fmt yuv420p washington_linear_32.mp4
ffmpeg -y -framerate 32 -i movies\washington_linear_128d_10h_cycle-%08d.png -c:v libx264 -pix_fmt yuv420p washington_linear_128.mp4
ffmpeg -y -framerate 32 -i movies\washington_linear_512d_10h_cycle-%08d.png -c:v libx264 -pix_fmt yuv420p washington_linear_512.mp4


python .\image_learn.py -i .\input\mona_lisa.jpg -t circular -e 2 -d 8  -n 10 --learning_rate .1 -p 3 -x 2 --save_frames movies --cycles 150
python .\image_learn.py -i .\input\mona_lisa.jpg -t circular -e 2 -d 32 -n 16 --learning_rate .1 -p 3 -x 2 --save_frames movies --cycles 150
python .\image_learn.py -i .\input\mona_lisa.jpg -t circular -e 3 -d 128 -n 24 --learning_rate .1 -p 3 -x 2 --save_frames movies --cycles 200
python .\image_learn.py -i .\input\mona_lisa.jpg -t circular -e 3 -d 512 -n 64 --learning_rate .1 -p 3 -x 2 --save_frames movies --cycles 200

