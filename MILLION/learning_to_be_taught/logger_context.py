import rlpyt.utils.logging.logger as logger
import torch
import os.path as osp
import subprocess
import json
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams


def filter_jsonable(input_dict: dict):
    output_dict = dict()
    for key, value in input_dict.items():
        if type(value) == dict:
            output_dict[key] = filter_jsonable(value)
        else:
            try:
                json.dumps(value)
                # if successful, no exception
                output_dict[key] = value
            except TypeError:
                # value is not jsonable
                pass

    return output_dict


def flatten_dict(input_dict):
    if type(input_dict) == dict:
        output_dict = dict()
        for key, value in input_dict.items():
            if type(value) == dict:
                for inner_key, inner_value in flatten_dict(value).items():
                    output_dict[str(key) + '/' + str(inner_key)] = inner_value
            else:
                output_dict[key] = value

        return output_dict
    else:
        return input_dict


def log_hparams(params):
    writer = logger.get_tf_summary_writer()
    params = flatten_dict(params)
    filtered_params = dict()
    for key, value in params.items():
        if type(value) in [int, float, str, bool]:# , torch.Tensor]:
            filtered_params[key] = value
    hparam_dict = filtered_params
    metric_dict = {'Return/Average': float('nan')}
    exp, ssi, sei = hparams(hparam_dict, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
    for k, v in metric_dict.items():
        writer.add_scalar(k, v)


def config_logger(log_dir='./bullet_data', name='run', log_params: dict = None, snapshot_mode="last"):
    logger.set_snapshot_mode(snapshot_mode)
    logger.set_log_tabular_only(False)

    run_ID = 0
    while osp.exists(osp.join(log_dir, f'{name}_{run_ID}')):
        run_ID += 1
    log_dir = osp.join(log_dir, f"{name}_{run_ID}")
    exp_dir = osp.abspath(log_dir)
    print('exp_dir: ' + exp_dir)

    tabular_log_file = osp.join(exp_dir, "progress.csv")
    text_log_file = osp.join(exp_dir, "debug.log")
    params_log_file = osp.join(exp_dir, "params.json")

    logger.set_snapshot_dir(exp_dir)
    logger.add_text_output(text_log_file)
    logger.add_tabular_output(tabular_log_file)
    logger.push_prefix(f"{name}_{run_ID} ")
    logger.set_tf_summary_writer(SummaryWriter(exp_dir))

    if log_params is None:
        log_params = dict()
    log_params["name"] = name
    log_params["run_ID"] = run_ID
    log_hparams(log_params)
    log_params = filter_jsonable(log_params)
    with open(params_log_file, "w") as f:
        json.dump(log_params, f)
        f.write('\n\nlast commit: ' + subprocess.getoutput('git log --max-count=1') + '\n\n')
        f.write(subprocess.getoutput('git diff'))
