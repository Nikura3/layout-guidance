import pathlib
import datetime
import csv

import numpy as np

import torch

class Logger:
    def __init__(self, path: pathlib.Path):
        self._name=""
        self._reserved_memory=0
        self._runs = []
        self._csvwriter = csv.writer(open(path + "performance_generation_log.csv", "w"))
        fields = ['ID', 'Desc', 'GPU', 'Reserved memory (GB)', 'Avg time (s)']
        self._csvwriter.writerow(fields)
    def log_gpu_memory_instance(self):
        self._name = torch.cuda.get_device_name(torch.cuda.current_device())
        self._reserved_memory = torch.cuda.memory_reserved(torch.cuda.current_device())

    def log_time_run(self,start,end):
        self._runs.append((start,end))

    def save_log_to_csv(self, prompt):
        all_elapsed=[]

        #for i in range(0,17):
        #    all_elapsed.append(np.nan)

        for run in enumerate(self._runs):
            #elapsed=run[1][1]-run[1][0]
            all_elapsed.append(run[1][1]-run[1][0])
            #all_elapsed[run[0]]=elapsed

        avg_elapsed = np.nanmean(all_elapsed)
        self._csvwriter.writerow([datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
                                 prompt,
                                 self._name,
                                 "{:.2f}".format(self._reserved_memory/pow(2,30)),
                                 "{:.2f}".format(avg_elapsed)])