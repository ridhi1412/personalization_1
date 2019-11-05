'''format a file'''
from yapf.yapflib.yapf_api import FormatFile  # reformat a file
FormatFile('common.py', in_place=True)
FormatFile('data_loader.py', in_place=True)
FormatFile('model1.py', in_place=True)
FormatFile('test3.py', in_place=True)