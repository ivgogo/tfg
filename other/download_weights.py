'''
Errors with pytorch weights downloads:
    https://github.com/pytorch/vision/issues/7744#issuecomment-1757321451
    https://github.com/pytorch/vision/pull/7898

    Torchvision releases:
    https://github.com/pytorch/vision/releases

    Torchvision repo:
    https://github.com/pytorch/vision?tab=readme-ov-file

Solution --> download and host weights locally or upgrade torchvision to > 0.16.0

'''

import torchvision.models

# list = torchvision.models.list_models() # weights for all models
list = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']   # weights for models of our interest

problems = []
for name in list:
    print(name)
    kwargs = {"quantize": True} if "quantized" in name else dict()
    for weights in torchvision.models.get_model_weights(name):
        print(weights)
        try:
            torchvision.models.get_model(name, weights=weights, **kwargs)
        except RuntimeError as e:
            problems.append((name, weights, e))
            print(e)

print("Following models need to be fixed:")
for name, weights, e in problems:
    print(name, weights, e)