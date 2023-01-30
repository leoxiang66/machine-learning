from itertools import product
import wandb
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

def hyperparameter_tune(*,
                        hyper_range: dict,
                        run_call: callable,
                        API_key,
                        project_name,
                        framework:str = 'Hugging Face'

                        ) -> list:
    '''
    hyperparameter tuning in sequential order with wandb visualization.
    Example:

    >>>
    hyper_range = dict(
        lr = [0.01,0.1],
        hd = [32,64]
    )

    >>>
    def train(hyper_config,run_name):
        print('Config: ',hyper_config)
        print('Run Name: ',run_name)

    >>>
    hyperparameter_tune(
        hyper_range=hyper_range,
        run_call=train,
        API_key='pseudo key',
        project_name='my project'
    )

    >>>
    Config:  (0.01, 32)
    Run Name:  ('lr', 0.01), ('hd', 32)
    Config:  (0.01, 64)
    Run Name:  ('lr', 0.01), ('hd', 64)
    Config:  (0.1, 32)
    Run Name:  ('lr', 0.1), ('hd', 32)
    Config:  (0.1, 64)
    Run Name:  ('lr', 0.1), ('hd', 64)

    :param framework: the framework used with wandb. Currently supported: Hugging Face
    :param project_name: wandb project name
    :param API_key: user API key for wandb
    :param hyper_range: dict object with keys being the name of hyperparameter, values being a list of hyperparameter values to explore
    :param run_call: callable object, should have 2 keyword arguments: "hyper_config". The first argument should be one hyperparameter configuration, e.g. a list of values; In this function call, the model should be trained on the given hyper config and return the result in dict.
    :return: return a list of hyper config run results
    '''


    if framework == 'Hugging Face':
        return wd__hf__(hyper_range, API_key, project_name, run_call)
    else:
        raise NotImplementedError('Other frameworks will be supported in the future.')



def wd__hf__(hyper_range, API_key, project_name, run_call):
    wandb.login(key=API_key)
    results = []

    for conf in product(*hyper_range.values()):
        wandb.init(project=project_name)
        run_name = zip(hyper_range.keys(), conf)
        run_name = str(list(run_name))[1:-1]
        wandb.run.name = run_name
        wandb.run.save()

        result = run_call(
            hyper_config=conf,
        )
        results.append(result)

        wandb.finish()

    return results

def tb_torch(
            hyper_range: dict,
            run_call: callable,
            project_name: str,
):
    results = []

    for conf in product(*hyper_range.values()):
        run_name = zip(hyper_range.keys(), conf)
        run_name = str(list(run_name))[1:-1]
        path = Path(project_name).joinpath(run_name)
        writer = SummaryWriter(log_dir=path.__str__())

        result = run_call(
            hyper_config=conf,
            writer=writer,
        )
        results.append(result)

    return results


def hyperparameter_tune_tensorboard(*,
                        hyper_range: dict,
                        run_call: callable,
                        project_name: str,
                        framework: str
                                ):
    if framework == 'pytorch':
        return tb_torch(hyper_range,run_call,project_name)
    else:
        raise NotImplementedError('Other frameworks will be supported in the future.')




