
import argparse
import os
import os.path as osp
import torch.utils.data
import numpy as np
import tqdm
from tqdm.contrib import tenumerate
from utils import load_data, Model
import pickle as pkl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    args = parser.parse_args()
    return args

def main(args):
    batch_size = args.batch_size
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)

    # data
    train_samples, test_samples = load_data('data1.mat')
    train_data = np.stack([sample['x'] for sample in train_samples])

    train_loader = torch.utils.data.DataLoader(
        train_samples, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_samples, batch_size=1, shuffle=True
    )


    # model
    model = Model(1, train_data=train_data)

    for epoch in range(1000):    
        for i, batch in tenumerate(train_loader):
            x = batch['x'].numpy()
            y = batch['y'].numpy()
            returns = model.get_loss(x, y)
            grad_l2 = model.sgd(x, y, returns, lr=0.01)
            loss = returns[0]
            # if i % 10 == 0:
                # print(f'step {i}: {loss.item()}')

            # pred = model.predict(x)
            # pred = pred > 0
            # print(pred.sum() / pred.shape[0])
            # gt = y > 0
            # print(f'acc: {100 * np.sum(gt == pred) / x.shape[0]: .02f}%')    

        num_correct, num_samples = 0, 0

        for i, batch in tenumerate(test_loader):
            x = batch['x'].numpy()
            y = batch['y'].numpy()
            pred = model.predict(x)
            pred = pred > 0
            gt = y > 0
            num_correct += np.sum(gt == pred)
            num_samples += y.shape[0]

        with open(osp.join(save_dir, f'weight_epoch{epoch}.pkl'), 'wb') as f:
            pkl.dump(model.weights, f)

        print(f'acc: {100 * num_correct / num_samples: .02f}%')    

        

if __name__ == "__main__":
    args = parse_args()
    main(args)