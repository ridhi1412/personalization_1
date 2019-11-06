'''format a file'''
from yapf.yapflib.yapf_api import FormatFile  # reformat a file
FormatFile('common.py', in_place=True)
FormatFile('data_loader.py', in_place=True)
FormatFile('als_model.py', in_place=True)
FormatFile('lightfm_model.py', in_place=True)
FormatFile('baseline_model.py', in_place=True)
FormatFile('nearest_neighbor_model.py', in_place=True)