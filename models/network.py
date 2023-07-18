import models.densenet2d as dense
import models.resnet as res
from models.tinynet import TinyNet
from models.vit import VisionTransformer

def network(mode="encoder",
            net="tiny",
            pretrained=False,
            n_layer=18,
            output_dim=128,
            num_classes=2,
            rep_dim=512,
            hidden_dim=256):

    print(net)

    if net == "densenet":
        output = dense.densenet121(pretrained=pretrained, mode=mode,output_dim=output_dim, num_classes=num_classes)
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
