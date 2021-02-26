
from modules import FCModule, Loss
from utils.common_utils import get_hidden_layer_sizes


def ae_wrapper(config):
    from models.auto_encoder import AutoEncoder

    # args
    input_size = config.input_size
    btl_size = config.btl_size
    n_layers = config.n_layers

    # input_size -> flatten
    if type(input_size) != int:
        C, H, W = input_size
        input_size = C * H * W
    else:
        input_size = input_size

    encoder = FCModule(
        input_size=input_size,
        output_size=btl_size,
        hidden_sizes=get_hidden_layer_sizes(input_size, btl_size, n_hidden_layers=n_layers - 1),
        use_batch_norm=True,
        act="leakyrelu",
        last_act=None,
    )

    decoder = FCModule(
        input_size=btl_size,
        output_size=input_size,
        hidden_sizes=get_hidden_layer_sizes(btl_size, input_size, n_hidden_layers=n_layers - 1),
        use_batch_norm=True,
        act="leakyrelu",
        last_act=None,
    )

    model = AutoEncoder(
        encoder=encoder,
        decoder=decoder,
        recon_loss=Loss("mse", reduction="sum"),
    )

    return model


def get_model(config):
    model = ae_wrapper(config)
    if config.gpu_id >= 0:
        model = model.cuda(config.gpu_id)

    return model