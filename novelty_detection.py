
from torch import optim
from ignite.engine import Engine, Events

from utils.metric import *
from utils.data_loaders import get_loaders, get_input_size
import pandas as pd


class NoveltyDetecter():

    def __init__(self, config):
        self.config = config

    def test(self,
             model,
             dset_manager,
             train_loader,
             valid_loader,
             test_loader,
             df_test):
        from reconstruction_aggregation import get_diffs

        model.eval()

        with torch.no_grad():
            _train_x, _ = dset_manager.get_transformed_data(train_loader)
            _valid_x, _ = dset_manager.get_transformed_data(valid_loader)
            _test_x, _test_y = dset_manager.get_transformed_data(test_loader)

            if self.config.unimodal_normal:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), False, True)
            else:
                _test_y = np.where(np.isin(_test_y, [self.config.target_class]), True, False)

            train_diff_on_layers = get_diffs(_train_x, model, batch_size=config.batch_size)
            valid_diff_on_layers = get_diffs(_valid_x, model)
            test_diff_on_layers = get_diffs(_test_x, model)

        from utils.metric import get_d_norm_loss, get_recon_loss

        _, base_auroc, base_aupr, base_f1scores, base_precisions, base_recalls = get_recon_loss(
            valid_diff_on_layers[0],
            test_diff_on_layers[0],
            _test_y,
            f1_quantiles=[.90],
        )


        _, sap_auroc, sap_aupr, sap_f1scores, sap_precisions, sap_recalls = get_d_loss(
            train_diff_on_layers,
            valid_diff_on_layers,
            test_diff_on_layers,
            _test_y,
            gpu_id=self.config.gpu_id,
            start_layer_index=self.config.start_layer_index,
            end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
            norm_type=2,
            f1_quantiles=[.90],
        )

        score, nap_auroc, nap_aupr, nap_f1scores, nap_precisions, nap_recalls = get_d_norm_loss(
            train_diff_on_layers,
            valid_diff_on_layers,
            test_diff_on_layers,
            _test_y,
            self.config,
            gpu_id=self.config.gpu_id,
            start_layer_index=self.config.start_layer_index,
            end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
            norm_type=2,
            f1_quantiles=[.90],
        )

        temp_df = pd.DataFrame(
            [{'base_auroc': base_auroc, 'sap_auroc': sap_auroc, 'nap_auroc': nap_auroc,
              'base_f1score': base_f1scores, 'sap_f1score': sap_f1scores, 'nap_f1score': nap_f1scores,
              'base_precision': base_precisions, 'sap_precision': sap_precisions, 'nap_precision': nap_precisions,
              'base_recalls': base_recalls, 'sap_recalls': sap_recalls, 'nap_recalls': nap_recalls,
              'base_aupr': base_aupr, 'sap_aupr': sap_aupr, 'nap_aupr': nap_aupr}])


        df_test = df_test.append(temp_df, ignore_index=True)

        return (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), df_test


    def train(self, model, train_loader, valid_loader):

        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        trainer = Engine(model.step)
        trainer.model, trainer.optimizer, trainer.config = model, optimizer, self.config
        trainer.train_history = []
        trainer.test_history = []

        evaluator = Engine(model.validate)
        evaluator.model, evaluator.config, evaluator.lowest_loss = model, self.config, np.inf
        evaluator.valid_history = []

        model.attach(trainer, evaluator, self.config)

        def run_validation(engine, evaluator, valid_loader):
            evaluator.run(valid_loader, max_epochs=1)

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, run_validation, evaluator, valid_loader
        )

        @trainer.on(Events.EPOCH_COMPLETED)
        def append_train_loss_history(engine):
            engine.train_history += [float(engine.state.metrics['recon'])]

        @evaluator.on(Events.EPOCH_COMPLETED)
        def append_valid_loss_history(engine):
            from copy import deepcopy
            loss = float(engine.state.metrics['recon'])
            if loss < engine.lowest_loss:
                engine.lowest_loss = loss
                engine.best_model = deepcopy(engine.model.state_dict())

            engine.valid_history += [loss]

        _ = trainer.run(train_loader, max_epochs=self.config.n_epochs)
        model.load_state_dict(evaluator.best_model)

        return trainer.train_history, evaluator.valid_history, trainer.test_history, model



def get_config():
    import argparse

    p = argparse.ArgumentParser()

    p.add_argument('--n_epochs', type=int, default=20)

    p.add_argument('--batch_size', type=int, default=7000)
    p.add_argument('--slicing_size', type=int, default=56000)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--data', type=str, default='hsr_objectdrop')
    p.add_argument('--unimodal_normal', action='store_true', default=False)
    p.add_argument('--target_class', type=str, default=1)

    p.add_argument('--novelty_ratio', type=float, default=.0)
    p.add_argument('--btl_size', type=int, default=100) # 100, 10
    p.add_argument('--n_layers', type=int, default=5) # 5, 3

    p.add_argument('--start_layer_index', type=int, default=0)
    p.add_argument('--end_layer_index', type=int, default=-1)
    p.add_argument('--from', type=str, default="youngjae")


    p.add_argument('--folder_name', type=str, default="hsr_objectdrop/")
    p.add_argument('--models', type=str, default='ae')
    p.add_argument('--save_mode', action='store_true', default=False)

    p.add_argument('--data_folder_name', type=str, default="/data_ssd/hsr_dropobject/")
    p.add_argument('--file_name', type=str, default="data_sum") # data_sum data_sum_motion, data_sum_free
    p.add_argument('--sensor', type=str, default="All")  # All hand_camera force_torque head_depth mic LiDAR
    p.add_argument('--saved_name', type=str, default="datasets/All_100.pt")
    p.add_argument('--saved_data', type=str, default="All")
    p.add_argument('--saved_result', type=str, default="1_26/All_sec")
    p.add_argument('--object_select_mode', action='store_true', default=False)
    p.add_argument('--train_diffs', type=str, default='datasets/All_train_diffs.pt')

    config = p.parse_args()

    if config.file_name is not 'data_sum':
        config.slicing_size = 7000

    return config


def main(config):
    from model_builder import get_model
    
    config.input_size = get_input_size(config)
    model = get_model(config)


    detecter = NoveltyDetecter(config)

    if config.verbose >= 1:
        print(config)

    if config.verbose >= 2:
        print(model)

    dset_manager, train_loader, valid_loader, test_loader = get_loaders(config)
    train_history, valid_history, test_history, model = detecter.train(model,
                                                                train_loader,
                                                                valid_loader)
    torch.save(model.state_dict(), config.saved_name)

    # testModel
    df_test = pd.DataFrame([{'base_auroc': 0, 'sap_auroc': 0, 'nap_auroc': 0, 'base_f1score':0, 'sap_f1score':0, 'nap_f1score': 0}])

    (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), df_test = detecter.test(
        model,
        dset_manager,
        train_loader,
        valid_loader,
        test_loader,
        df_test)

    # df_test[1:].to_csv('/data_ssd/hsr_dropobject/result_csv/'+config.saved_result+'.csv')

    return (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr)


if __name__ == '__main__':
    config = get_config()
    start = time.time()
    
    (base_auroc, base_aupr), (sap_auroc, sap_aupr), (nap_auroc, nap_aupr), = main(config)
    end = time.time()

    print((end - start)/60) # min
    print('BASE AUROC: %.4f AUPR: %.4f' % (base_auroc, base_aupr))

    print('RaPP SAP AUROC: %.4f AUPR: %.4f' % (sap_auroc, sap_aupr))
    print('RaPP NAP AUROC: %.4f AUPR: %.4f' % (nap_auroc, nap_aupr))
