import os 

#os.chdir('./VITS-fast-fine-tuning')
if not os.path.exists("OUTPUT_MODEL"):
    os.system('mkdir OUTPUT_MODEL')
    os.chdir('./OUTPUT_MODEL')
    os.system('gdown https://drive.google.com/uc?id=1vQeqn_LApY5j69O82Ihd0JeyGeOP6iiz')
    os.system('gdown https://drive.google.com/uc?id=18N4sDaWbPWoAlxfAhWM0GU5fcp-ynPqK')
    os.chdir('..')

#os.system('pip uninstall gradio')
#os.system('pip install gradio==3.50.2') # 4.31.4
#os.system('cp ./configs/modified_finetune_speaker.json ./finetune_speaker.json')
os.system('python VC_inference.py --model_dir ./OUTPUT_MODEL/G_latest.pth --share True')