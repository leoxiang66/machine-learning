# if __name__ == '__main__':
#     from supervised_learning.models.wavenet import WaveNet, WaveNetForClassification
#     from torchinfo import summary
#     import torch
#     a = WaveNet(5,7,7,500)
#     import torch.nn as nn
#     import numpy as np
#     # summary(a)
#
#     # named_layers = list(a.named_children())  # 所有的layers
#     # for i in range(len(named_layers)):
#     #     print(f'''第{i + 1}个layer: {named_layers[i]}''')
#
#
#
#     wavenet_classification = WaveNetForClassification(5,7,7,10,500)
#     torch.manual_seed(0)
#     X = torch.randn(100,7,500)
#     y = torch.randint(0,10,size=(100,))
#     x = a(X)
#     x2 = wavenet_classification(X)
#
#     # 创建网络模型
#     tudui = wavenet_classification
#
#     # 损失函数
#     loss_fn = nn.CrossEntropyLoss()
#
#     # 优化器
#     learning_rate = 1e-2
#     optimizer = torch.optim.Adam(tudui.parameters(), lr=learning_rate)
#
#     # 训练的轮数
#     epoch = 50
#
#     train_epoch_loss = []
#     valid_epoch_loss = []
#
#
#
#     for i in range(epoch):
#         print("-------第 {} 轮训练开始-------".format(i + 1))
#
#         # 训练步骤开始
#         tudui.train()
#         train_step_loss = []
#         train_acc = 0
#         step = 0
#
#
#
#         outputs = tudui(X)  # 求模型的输出
#         optimizer.zero_grad()  # 梯度清零
#         loss = loss_fn(outputs, y)  # 求loss
#
#         accuracy = (outputs.argmax(1) == y).sum()
#         train_acc += accuracy
#
#         train_step_loss.append(loss.item())
#         step += 1
#         if (step % 100 == 0):
#             print(f'第{i + 1}轮第{step}训练step时的loss: {loss.item()}')
#
#         # 优化器优化模型
#         loss.backward()  # 求梯度
#         optimizer.step()  # 更新参数
#         train_epoch_loss.append(np.average(train_step_loss))
#
#
#
#         print(f"第{i + 1}轮整体训练集上的Loss: {train_epoch_loss[-1]}")
#         print(f"第{i + 1}轮整体训练集上的正确率: {(train_acc / len(X))}")
#
#
#

if __name__ == '__main__':
    import json
    from optimization import hyperparameter_tune

    with open("config.json", 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    import numpy as np

    hyper_range = dict(
        lr = [0.01,0.1],
        hd = [32,64]
    )

    def train(hyper_config):
        print('Config: ',hyper_config)
        epoch = 10
        return dict(
            train = dict(
                accuracy = np.random.randn(epoch),
                loss = np.random.randint(0,100,epoch)
            ),
            val = dict(
                accuracy=np.random.randn(epoch),
                loss=np.random.randint(0, 100, epoch)
            )
        )


    hyperparameter_tune(
        hyper_range=hyper_range,
        run_call=train,
        API_key=load_dict['key'],
        project_name='my project'
    )

