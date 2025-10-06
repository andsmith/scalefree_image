python .\nnet_test.py nnet_barn_b6 -i .\input\barn.png -l 16 32 -b 16 -e 1 -t 65536 -r .01 -x 2 -k 200 --clobber
python .\nnet_test.py nnet_barn_b64k -i .\input\barn.png -l 16 32 -b 65536 -e 100 -t 65536 -r .01 -x 2  -k 200 --clobber


python .\nnet_test.py nnet_barn_64_b16 -i .\input\barn.png -l 64 128 -b 16 -e 1 -t 65536 -r .01 -x 2 -k 150  --clobber
python .\nnet_test.py nnet_barn_64_b64k -i .\input\barn.png -l 64 128 -b 65536 -e 50 -t 65536 -r .01 -x 2 -k 150  --clobber

python .\nnet_test.py nnet_barn_256_b16 -i .\input\barn.png -l 256 512 -b 16 -e 1 -t 65536 -r .01 -x 2  -k 100 --clobber
python .\nnet_test.py nnet_barn_256_b64k -i .\input\barn.png -l 256 512 -b 65536 -e 30 -t 65536 -r .01 -x 2  -k 100 --clobber
