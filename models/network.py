import models.densenet2d as dense
import models.resnet as res
from models.tinynet import TinyNet
from models.vit import VisionTransformer

def network(mode: str = "encoder",
            net: str = "tiny",
            pretrained: bool = False,
            n_layer: int = 18,
            num_classes: int = 4,
            rep_dim: int = 256,
            hidden_dim: int = 128,
            output_dim: int = 64):

    """Function to call the selected network by the user.
    :param mode: select if encore, representation or classification mode.
    :param net: select if densenet, resnet, tiny or vit. Encoder type.
    :param pretrained: True if pretrained on ImageNet (works only for ResNet).
    :param n_layer: number of layers in the ResNet encoder.
    :param num_classes: number of classes if classification mode.
    :param rep_dim: representation space dimension of the encoder.
    :param hidden_dim: hidden space dimension of the encoder.
    :param output_dim: output space dimension of the encoder."""



    if net == "densenet":
        output = dense.densenet121(pretrained=pretrained,
                                   mode=mode,
                                   output_dim=output_dim,
                                   num_classes=num_classes)

    elif net == "resnet":

        if n_layer == 18:
            output = res.resnet18(pretrained=pretrained,
                                  mode=mode,
                                  output_dim=output_dim,
                                  num_classes=num_classes,
                                  rep_dim=rep_dim,
                                  hidden_dim=hidden_dim)
        if n_layer == 34:
            output = res.resnet34(pretrained=pretrained,
                                  mode=mode,
                                  output_dim=output_dim,
                                  num_classes=num_classes,
                                  rep_dim=rep_dim,
                                  hidden_dim=hidden_dim)
        if n_layer == 50:
            output = res.resnet50(pretrained=pretrained,
                                  mode=mode,
                                  output_dim=output_dim,
                                  num_classes=num_classes,
                                  rep_dim=rep_dim,
                                  hidden_dim=hidden_dim)

    elif net == "tiny":
        output = TinyNet(pretrained=pretrained,
                         num_classes=num_classes,
                         mode=mode,
                         rep_dim=rep_dim,
                         hidden_dim=hidden_dim,
                         output_dim=output_dim)

    elif net == "vit":
        output = VisionTransformer()

    return output
