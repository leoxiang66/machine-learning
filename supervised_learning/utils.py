from sklearn.model_selection import train_test_split
import torch

def k_fold_cross_validation(k: int, X:torch.Tensor, y:torch.Tensor ,train_fn:callable ,compute_val_loss_fn:callable):
    '''

    :param k: number of folds
    :param X: the whole dataset of features.
    :param y: the whole dataset of labels
    :param train_fn: the function takes two inputs: train_x and train_y, returns None. In this function the model should be trained on train_x and train_y
    :param compute_val_loss_fn: the function takes two inputs: val_x and val_y, which is the validation dataset, and it should return a loss scalar
    :return:
    '''

    # k等分dataset
    assert k>1
    tmp_x = X
    tmp_y = y
    batches = []

    for i in [1/z for z in range(k,1,-1)]:
        tmp_x, X_test, tmp_y, y_test = train_test_split(
        tmp_x, tmp_y, test_size=i, random_state=0)

        batches.append((X_test,y_test))
    batches.append((tmp_x,tmp_y))

    # 计算k次
    val_losses = []
    for j in range(k):
        val_x, val_y = batches[j]
        train_indices = [i for i in range(k) if i != j]

        ## training
        train_x = torch.concat([batches[x][0] for x in train_indices])
        train_y = torch.concat([batches[x][1] for x in train_indices])
        train_fn(train_x,train_y)


        ## validation loss
        val_losses.append(compute_val_loss_fn(val_x,val_y))

    return torch.mean(torch.Tensor(val_losses))


