from itertools import product
import os

def hyperparameter_tune(*,
                        hyper_range: dict,
                        run_call: callable,
                        API_key,
                        project_name
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

    :param project_name: wandb project name
    :param API_key: user API key for wandb
    :param hyper_range: dict object with keys being the name of hyperparameter, values being a list of hyperparameter values to explore
    :param run_call: callable object, should have 2 keyword arguments: "hyper_config" and "run_name". The first argument should be one hyperparameter configuration, e.g. a list of values; The second is current run_name. In this function call, the model should be trained on the given hyper config and store the result (either uploaded to wandb or returned)
    :return: return a list of hyper config run results
    '''

    os.system(f'wandb login {API_key}')
    os.environ['WANDB_PROJECT'] = project_name
    print(f'''Succesfully set up the project name: {os.environ['WANDB_PROJECT']}''')

    results = []
    for conf in product(*hyper_range.values()):
        run_name = zip(hyper_range.keys(), conf)
        run_name = str(list(run_name))[1:-1]

        results.append(run_call(
            hyper_config = conf,
            run_name = run_name,
        ))
    return results

if __name__ == '__main__':
    hyper_range = dict(
        lr = [0.01,0.1],
        hd = [32,64]
    )

    def train(hyper_config,run_name):
        print('Config: ',hyper_config)
        print('Run Name: ',run_name)


    hyperparameter_tune(
        hyper_range=hyper_range,
        run_call=train,
        API_key='pseudo key',
        project_name='my project'
    )