import requests
import json
import pandas as pd
import numpy as np


class WebserviceConfig:
    def __init__(self, config):
        self.config = config

    def get_parameters(self, args):
        # بررسی، دریافت و مقداردهی اولیه به پارامترها
        parameters = {}
        if args.get('eos') is not None:
            parameters["eps"] = float(args.get('eps'))
        else:
            parameters["eps"] = self.config['eps']

        if args.get('min_samples') is not None:
            parameters["min_samples"] = int(args.get('min_samples'))
        else:
            parameters["min_samples"] = self.config['min_samples']

        if args.get('prune_thresh') is not None:
            parameters["prune_thresh"] = float(args.get('prune_thresh'))
        else:
            parameters["prune_thresh"] = self.config['prune_thresh']

        if args.get('noise_deletion') is not None:
            parameters["noise_deletion"] = args.get('noise_deletion')
        else:
            parameters["noise_deletion"] = self.config['noise_deletion']

        if args.get('sim_thresh') is not None:
            parameters["sim_thresh"] = float(args.get('sim_thresh'))
        else:
            parameters["sim_thresh"] = self.config['sim_thresh']

        return parameters

    def validate_input(self, data):
        # بررسی صحت پارامتر های بدنه
        # شامل فیلدهای مورد نظر باشد
        for idx, msg in enumerate(data["messages"]):
            for col in self.config['data_columns']:
                if not col in msg:
                    return False, f'record {idx} does not have {col}!'

        return True, None
