python3 -m fastchat.serve.controller > controller.log 2 >&1  

python3 -m fastchat.serve.model_worker --model-path baichuan-inc/baichuan-7B > model_worker.log 2>&1 

python3 -m fastchat.serve.test_message --model-name baichuan-inc/baichuan-7B > test_message.log 2>&1