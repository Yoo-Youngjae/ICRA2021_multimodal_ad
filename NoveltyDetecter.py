
import numpy as np
import torch
from matplotlib import pyplot as plt
from utils.normalize import Standardizer, Rotater

class NoveltyDetecter():

    def __init__(self, config):
        self.config = config

    def show_detect_slip(self, test_x, test_y, model): # test_x length = 30
        model.eval()

        if isinstance(test_x, np.ndarray):
            test_x = torch.tensor(test_x)
        test_x = test_x.view(-1,1,self.config.input_size)
        loss_fn = torch.nn.MSELoss(reduction="sum")
        x_plot = []
        y_plot = []

        true_plot = []
        false_plot = []

        label_plot = []
        for i, (x, y) in enumerate(zip(test_x, test_y)):
            model.eval()
            x = x.to(next(model.parameters()).device).float()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            print(i, loss.item(), y)
            x_plot.append(i)
            y_plot.append(loss.item())
            if bool(y) == True: # is np true
                label_plot.append(1)
                true_plot.append(x)
            else:
                label_plot.append(0)
                false_plot.append(x)

        plt.subplot(2,1,1)
        plt.plot(x_plot, y_plot)
        plt.title('RaPP Loss')

        plt.subplot(2, 1, 2)
        plt.plot(x_plot, label_plot)
        plt.title('Label')
        plt.tight_layout()
        plt.show()


    def test(self,
             model,
             test_loader,
             config
             ):
        from reconstruction_aggregation import get_diffs

        model.eval()

        with torch.no_grad():

            _test_x, _test_y = self.get_transformed_data(test_loader)

            # _test_x = _test_x.view(171,1,2048)[0]
            # _test_y = _test_y[0]

            _test_y = np.where(np.isin(_test_y, [self.config.target_class]), True, False)

            show_mode = False
            if show_mode:
                self.show_detect_slip(_test_x, _test_y, model)

            test_diff_on_layers = get_diffs(_test_x, model)

        score = self.get_d_norm_loss(
            test_diff_on_layers,
            config,
            gpu_id=self.config.gpu_id,
            start_layer_index=self.config.start_layer_index,
            end_layer_index=self.config.n_layers + 1 - self.config.end_layer_index,
            norm_type=2
        )

        return score

    def get_transformed_data(self, data_loader):
        """
        Multi indexing support
        """
        x = []
        y = []
        for i in data_loader.sampler:
            # indexing -> __getitem__ -> applying transform
            _x, _y = data_loader.dataset[i]
            x.append(_x)
            y.append(_y)

        if type(_x) == np.ndarray:
            x = np.stack(x)
        elif type(_x) == torch.Tensor:
            x = torch.stack(x)
        else:
            raise NotImplementedError

        if type(_y) == np.ndarray:
            y = np.array(y)
        elif type(_y) == torch.Tensor:
            y = torch.tensor(y)

        return x, y

    def get_d_norm_loss(self, test_diffs,
                        config,
                        start_layer_index=0,
                        end_layer_index=None,
                        gpu_id=-1,
                        norm_type=2
                        ):

        if end_layer_index is None:
            end_layer_index = len(test_diffs) + 1

        if start_layer_index > len(test_diffs) - 1:
            start_layer_index = len(test_diffs) - 1

        if end_layer_index - start_layer_index < 1:
            end_layer_index = start_layer_index + 1

        train_diffs = torch.load(config.train_diffs)

        test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]],
                               dim=-1).numpy()

        # |test_diffs| = (batch_size, dim * config.n_layers)

        rotater = Rotater()
        stndzer = Standardizer()

        rotater.fit(train_diffs, gpu_id=gpu_id)
        stndzer.fit(rotater.run(train_diffs, gpu_id=gpu_id))

        test_rotateds = stndzer.run(rotater.run(test_diffs, gpu_id=gpu_id))
        score = self.get_norm(test_rotateds, norm_type).mean(axis=1)

        return score

    def get_norm(self, x, norm_type=2):
        return abs(x) ** norm_type