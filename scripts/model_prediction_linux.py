import rdkit
from rdkit import Chem
from GraphConstructor import graph_from_mol_for_prediction
from MyUtils import *
torch.backends.cudnn.benchmark = True
import warnings
warnings.filterwarnings('ignore')
import torch
from MyModel import ModifiedChargeModelV2, ModifiedChargeModelV2New
import dgl
import argparse
import datetime


# Get the upper-level directory
path_marker = '/'
home_path = os.path.abspath(os.path.dirname(os.getcwd()))
model_home_path = home_path + path_marker + 'model_save'
pred_res_path = home_path + path_marker + 'outputs'
input_data_path = home_path + path_marker + 'inputs'
batch_size = 500
models = ['e4_3D_0_resp_.pth', 'e78_3D_1_resp_.pth', '2021-07-09_16_45_29_775301.pth']
charges = ['e4', 'e78', 'resp']


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--job_of_name', type=str, default='hello_charge', help="the unique flag for each job")
    argparser.add_argument('--type_of_charge', type=str, default='e4', help="type of charge, only support 'e4', 'e78', 'resp'")
    argparser.add_argument('--input_file', type=str, default='3cl-min.sdf', help="input file name, only support .sdf format")
    argparser.add_argument('--correct_charge', action="store_true", help="correct the summation of predicted charge of atoms in same "
                                                                         "molecule to make it as an integer or not")
    argparser.add_argument('--device', type=str, default='gpu', help="what type of device was used in the prediction, gpu or cpu")

    args = argparser.parse_args()
    job_of_name, type_of_charge, input_file, correct_charge = args.job_of_name, args.type_of_charge, args.input_file, args.correct_charge
    device = args.device
    cmdline = 'rm -rf %s && mkdir %s' % (pred_res_path + path_marker + job_of_name, pred_res_path + path_marker + job_of_name)
    os.system(cmdline)
    pred_res_path = pred_res_path + path_marker + job_of_name

    print('********************************job_of_name:%s, start*************************************************\n' % job_of_name)
    print('time', datetime.datetime.now())
    print(args)
    assert (type_of_charge in charges), "type of charge error, only support 'e4', 'e78' or 'resp'"
    assert input_file.endswith('.sdf'), "input file format error, only support .sdf format"
    assert os.path.exists(input_data_path + path_marker + input_file), 'input file %s not exists' % (input_data_path + path_marker + input_file)
    assert device in ['gpu', 'cpu'], 'only gpu or cpu was supported for device'

    data_folds = [input_file]
    type_of_charges = [type_of_charge]

    for data_fold_th, data_fold in enumerate(data_folds):
        for charge_th, charge in enumerate(type_of_charges):
            torch.cuda.empty_cache()
            # get the prediction data-sets:
            sdfs = Chem.SDMolSupplier(input_data_path + path_marker + data_fold, removeHs=False)
            valid_mol_ids = []
            graphs = []
            for i, mol in enumerate(sdfs):
                if mol:
                    try:
                        g = graph_from_mol_for_prediction(mol)
                        graphs.append(g)
                        valid_mol_ids.append(i)
                    except:
                        pass
                else:
                    pass

            # model prediction
            if charge != 'resp':
                ChargeModel = ModifiedChargeModelV2(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                    graph_feat_size=200,
                                                    dropout=0.1)
            else:
                ChargeModel = ModifiedChargeModelV2New(node_feat_size=36, edge_feat_size=21, num_layers=6,
                                                       graph_feat_size=200,
                                                       dropout=0.1, n_tasks=1 + 65)

            if device == 'gpu' and torch.cuda.is_available():
                # get the gpu device with maximum video memory
                outputs = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
                memory_gpu = [int(x.split()[2]) for x in outputs.readlines()]
                gpu_id = str(np.argmax(memory_gpu))
                device = torch.device("cuda:%s" % gpu_id)
            elif device == 'gpu' and not torch.cuda.is_available():
                print('no gpu device was available, the device was set as cpu')
                device = torch.device("cpu")
            else:
                device = torch.device("cpu")
            ChargeModel.load_state_dict(
                torch.load(model_home_path + path_marker + models[charges.index(type_of_charge)], map_location='cpu')['model_state_dict'])
            ChargeModel.to(device)

            ChargeModel.eval()
            with torch.no_grad():
                pred = []
                num_batch = len(graphs) // batch_size
                for i_batch in range(num_batch):
                    bg = dgl.batch(graphs[batch_size * i_batch:batch_size * (i_batch + 1)])
                    bg = bg.to(device)
                    feats = bg.ndata.pop('h')
                    efeats = bg.edata.pop('e')
                    outputs = ChargeModel(bg, feats, efeats)
                    if charge != 'resp':
                        pred.append(outputs.data.cpu().numpy())
                    else:
                        pred.append(outputs[:, 0].view(-1, 1).data.cpu().numpy())

                # last batch
                bg = dgl.batch(graphs[batch_size * num_batch:])
                bg = bg.to(device)
                feats = bg.ndata.pop('h')
                efeats = bg.edata.pop('e')
                outputs = ChargeModel(bg, feats, efeats)
                if charge != 'resp':
                    pred.append(outputs.data.cpu().numpy())
                else:
                    pred.append(outputs[:, 0].view(-1, 1).data.cpu().numpy())
                pred = np.concatenate(np.array(pred), 0)
            pred = iter(pred)

            sdf_file_name = '%s_new_%s.sdf' % (data_fold[:-4], charge)
            output_sdf_file = pred_res_path + path_marker + '%s_new_%s.sdf' % (data_fold[:-4], charge)
            writer = Chem.SDWriter(output_sdf_file)
            for valid_idx in valid_mol_ids:
                mol = sdfs[valid_idx]
                num_atoms = mol.GetNumAtoms()
                mol_pred_charges = []
                for i in range(num_atoms):
                    mol_pred_charges.append(next(pred)[0])

                # charge correlation
                sum_abs_pred = np.sum(np.abs(mol_pred_charges))
                dQ = np.sum(mol_pred_charges) - Chem.rdmolops.GetFormalCharge(mol)
                Qcorr = np.array(mol_pred_charges) - (np.abs(mol_pred_charges) * dQ) / sum_abs_pred

                if correct_charge:
                    for i in range(num_atoms):
                        mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(Qcorr[i]))
                    mol.SetProp('charge', str(int(np.sum(Qcorr))))
                else:
                    for i in range(num_atoms):
                        mol.GetAtomWithIdx(i).SetProp('molFileAlias', str(mol_pred_charges[i]))
                    mol.SetProp('charge', str(int(np.sum(Qcorr))))
                writer.write(mol)

            print('the predicted charge was stored in file: %s' % output_sdf_file)
            print('for assessing the predicted charge, please see the script (example):')
            print(home_path + path_marker + 'scripts' + path_marker + 'get_sdf_charge.py')
    print('********************************job_of_name:%s, end*************************************************\n' % job_of_name)
