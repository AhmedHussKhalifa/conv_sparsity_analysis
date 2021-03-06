rm ../gen/*.info*
echo 'Removed all info files!'
ls ../gen/*.info*/
echo 'No files - I know ;)'

time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name IV1
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name IV3
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name IV4
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name MobileNet
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name MobileNetV2
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name AlexNet
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name ResNet-V2-50
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name ResNet-V2-101
time /usr/bin/python3 run_inference.py --select Org --END 2 --model_name Pnasnet_Large
