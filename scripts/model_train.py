import rdkit
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from GraphConstructor import *
import time
from MyUtils import *
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
from scipy.stats import pearsonr

torch.backends.cudnn.benchmark = True
import warnings

warnings.filterwarnings('ignore')
import torch
from dgl.data import split_dataset
from MyModel import ModifiedChargeModelV2, ModifiedChargeModelV2New
from itertools import accumulate
from dgl.data import Subset
import os
import argparse


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def run_a_train_epoch(model, loss_fn, train_dataloader, optimizer, device, model_name):
    # training model for one epoch
    model.train()
    for i_batch, batch in enumerate(train_dataloader):
        model.zero_grad()
        bg, keys = batch
        bg = bg.to(device)
        Ys = bg.ndata.pop('charge')
        feats = bg.ndata.pop('h')
        efeats = bg.edata.pop('e')
        acsfs = bg.ndata.pop('acsf')
        outputs = model(bg, feats) if model_name == 'gat' else model(bg, feats, efeats)
        loss1 = loss_fn(outputs[:, 0].view(-1, 1), Ys).mean()
        loss2 = loss_fn(outputs[:, 1:], acsfs).mean()
        loss = loss1 + 20 * loss2
        loss.backward()
        optimizer.step()


def run_a_eval_epoch(model, validdataloader, device, model_name):
    true = []
    pred = []
    model.eval()
    with torch.no_grad():
        for i_batch, batch in enumerate(validdataloader):
            # ChargeModel.zero_grad()
            bg, keys = batch
            bg = bg.to(device)
            Ys = bg.ndata.pop('charge')
            feats = bg.ndata.pop('h')
            efeats = bg.edata.pop('e')
            outputs = model(bg, feats) if model_name == 'gat' else model(bg, feats, efeats)
            true.append(Ys.data.cpu().numpy())
            pred.append(outputs[:, 0].view(-1, 1).data.cpu().numpy())
    return true, pred


test_scripts = False


home_path = '/home/jike/dejunjiang/charge_prediction'
model_home_path = '/home/jike/dejunjiang/charge_prediction/model_save'
pred_res_path = '/home/jike/dejunjiang/charge_prediction/outputs'
input_data_path = '/home/jike/dejunjiang/charge_prediction/inputs'
path_marker = '/'
sdf_path = ' '


with open('/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/SuperAtmoicCharge/data/e4_valid_keys.pkl', 'rb') as f:
    data_keys = pickle.load(f)


sdf_files = [sdf_path + path_marker + key for key in data_keys]
print(sdf_files[0:3])
print(len(data_keys))
# remove the '.sdf' suffix for data_keys
data_keys = [key.strip('.sdf') for key in data_keys]


if test_scripts:
    sdf_files = sdf_files[:200]
    data_keys = data_keys[:200]


dataset = GraphDatasetNew(data_dirs=sdf_files, data_keys=data_keys, cache_bin_file='/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/SuperAtmoicCharge/data/e4_new.bin',
                          tmp_cache_path='/apdcephfs/private_dejunjiang/105/dejunjiang/wspy/SuperAtmoicCharge/data/tmpfiles', path_marker='/', num_process=16)

num_data = len(dataset)
print(num_data)
frac_list = [0.8, 0.1, 0.1]
frac_list = np.asarray(frac_list)
lengths = (num_data * frac_list).astype(int)
lengths[-1] = num_data - np.sum(lengths[:-1])
indices = np.random.RandomState(seed=43).permutation(num_data)
train_idx, valid_idx, test_idx = [indices[offset - length:offset] for offset, length in
                                  zip(accumulate(lengths), lengths)]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # model training parameters
    argparser.add_argument('--gpuid', type=str, default='0', help="gpu id for training model")
    argparser.add_argument('--lr', type=float, default=10 ** -3.0, help="Learning rate")
    argparser.add_argument('--epochs', type=int, default=50, help="Number of epochs in total")
    argparser.add_argument('--batch_size', type=int, default=200, help="Batch size")
    argparser.add_argument('--tolerance', type=float, default=0.0, help="early stopping tolerance")
    argparser.add_argument('--patience', type=int, default=50, help="early stopping patience")
    argparser.add_argument('--l2', type=float, default=10 ** -6, help="L2 regularization")
    argparser.add_argument('--repetitions', type=int, default=3, help="the number of independent runs")
    argparser.add_argument('--type_of_charge', type=str, default='e4',
                           help="type of charge, only support 'e4', 'e78', 'resp'")
    argparser.add_argument('--input_file', type=str, default='3cl-min_new_e4.sdf',
                           help="input file name, only support .mol2 or .sdf format")
    args = argparser.parse_args()
    gpuid, lr, epochs, batch_size, tolerance, patience, l2, repetitions = args.gpuid, args.lr, args.epochs, args.batch_size, args.tolerance, args.patience, \
                                                                          args.l2, args.repetitions
    type_of_charge, input_file = args.type_of_charge, args.input_file
    model_name = args.model_name

    stat_res = []
    for repetition_th in range(repetitions):
        torch.cuda.empty_cache()
        dt = datetime.datetime.now()
        filename = home_path + path_marker + 'model_save/{}_{:02d}_{:02d}_{:02d}_{:d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond)
        print('Independent run %s' % repetition_th)
        print('model file %s' % filename)
        seed_torch(repetition_th)
        if test_scripts:
            train_dataset = Subset(dataset, list(range(200)))
            valid_dataset = Subset(dataset, list(range(200)))
            test_dataset = Subset(dataset, list(range(200)))
            epochs = 5

        train_dataset = Subset(dataset, train_idx)
        valid_dataset = Subset(dataset, valid_idx)
        test_dataset = Subset(dataset, test_idx)

        print('number of train data:', len(train_dataset))
        print('number of valid data:', len(valid_dataset))
        print('number of test data:', len(test_dataset))
        train_dataloader = DataLoaderX(train_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)
        valid_dataloader = DataLoaderX(valid_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)
        test_dataloader = DataLoaderX(test_dataset, batch_size, shuffle=True, collate_fn=collate_fn_new)

        if charge != 'resp':
            ChargeModel = ModifiedChargeModelV2(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                graph_feat_size=200,
                                                dropout=0.1)
        else:
            ChargeModel = ModifiedChargeModelV2New(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                   graph_feat_size=200,
                                                   dropout=0.1, n_tasks=1 + 65)

        if repetition_th == 0:
            print(ChargeModel)
        device = torch.device("cuda:%s" % gpuid if torch.cuda.is_available() else "cpu")
        ChargeModel.to(device)
        optimizer = torch.optim.Adam(ChargeModel.parameters(), lr=lr, weight_decay=l2)
        stopper = EarlyStopping(mode='lower', patience=patience, tolerance=tolerance,
                                filename=filename)
        loss_fn = nn.MSELoss(reduction='none')

        for epoch in range(epochs):
            st = time.time()
            # train
            run_a_train_epoch(ChargeModel, loss_fn, train_dataloader, optimizer, device, model_name)

            # test
            train_true, train_pred = run_a_eval_epoch(ChargeModel, train_dataloader, device, model_name)
            valid_true, valid_pred = run_a_eval_epoch(ChargeModel, valid_dataloader, device, model_name)

            train_true = np.concatenate(np.array(train_true), 0)
            train_pred = np.concatenate(np.array(train_pred), 0)

            valid_true = np.concatenate(np.array(valid_true), 0)
            valid_pred = np.concatenate(np.array(valid_pred), 0)

            train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
            valid_rmse = np.sqrt(mean_squared_error(valid_true, valid_pred))

            early_stop = stopper.step(valid_rmse, ChargeModel)
            end = time.time()
            if early_stop:
                break
            print("epoch:%s \t train_rmse:%.4f \t valid_rmse:%.4f \t time:%.3f s" % (
                epoch, train_rmse, valid_rmse, end - st))

        # load the best model
        stopper.load_checkpoint(ChargeModel)

        train_true, train_pred = run_a_eval_epoch(ChargeModel, train_dataloader, device, model_name)
        valid_true, valid_pred = run_a_eval_epoch(ChargeModel, valid_dataloader, device, model_name)
        test_true, test_pred = run_a_eval_epoch(ChargeModel, test_dataloader, device, model_name)

        # metrics
        train_true = np.concatenate(np.array(train_true), 0).flatten()
        train_pred = np.concatenate(np.array(train_pred), 0).flatten()
        valid_true = np.concatenate(np.array(valid_true), 0).flatten()
        valid_pred = np.concatenate(np.array(valid_pred), 0).flatten()
        test_true = np.concatenate(np.array(test_true), 0).flatten()
        test_pred = np.concatenate(np.array(test_pred), 0).flatten()
        train_rmse, train_r2, train_mae, train_rp = np.sqrt(mean_squared_error(train_true, train_pred)), \
                                                    r2_score(train_true, train_pred), \
                                                    mean_absolute_error(train_true, train_pred), pearsonr(train_true,
                                                                                                          train_pred)
        valid_rmse, valid_r2, valid_mae, valid_rp = np.sqrt(
            mean_squared_error(valid_true, valid_pred)), \
                                                    r2_score(valid_true, valid_pred), \
                                                    mean_absolute_error(valid_true,
                                                                        valid_pred), pearsonr(
            valid_true, valid_pred)
        test_rmse, test_r2, test_mae, test_rp = np.sqrt(mean_squared_error(test_true, test_pred)), \
                                                r2_score(test_true, test_pred), \
                                                mean_absolute_error(test_true, test_pred), pearsonr(test_true,
                                                                                                    test_pred)
        print('***best %s charge model***' % model_name)
        print("train_rmse:%.4f \t train_r2:%.4f \t train_mae:%.4f \t train_rp:%.4f" % (
            train_rmse, train_r2, train_mae, train_rp[0]))
        print("valid_rmse:%.4f \t valid_r2:%.4f \t valid_mae:%.4f \t valid_rp:%.4f" % (
            valid_rmse, valid_r2, valid_mae, valid_rp[0]))
        print("test_rmse:%.4f \t test_r2:%.4f \t test_mae:%.4f \t test_rp:%.4f" % (
        test_rmse, test_r2, test_mae, test_rp[0]))
        stat_res.append([repetition_th, 'train', train_rmse, train_r2, train_mae, train_rp[0]])
        stat_res.append([repetition_th, 'valid', valid_rmse, valid_r2, valid_mae, valid_rp[0]])
        stat_res.append([repetition_th, 'test', test_rmse, test_r2, test_mae, test_rp[0]])
    stat_res_pd = pd.DataFrame(stat_res, columns=['repetition', 'group', 'rmse', 'r2', 'mae', 'rp'])
    stat_res_pd.to_csv(
        home_path + path_marker + 'stats' + path_marker + '{}_{:02d}_{:02d}_{:02d}_{:d}.csv'.format(
            dt.date(), dt.hour, dt.minute, dt.second, dt.microsecond), index=False)
    print(stat_res_pd[stat_res_pd.group == 'train'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'train'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'valid'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'valid'].std().values[-4:])
    print(stat_res_pd[stat_res_pd.group == 'test'].mean().values[-4:],
          stat_res_pd[stat_res_pd.group == 'test'].std().values[-4:])
