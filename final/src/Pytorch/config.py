class DefaultConfigs(object):
    train_data = "../total_train/" # train data
    test_data = "../test/"   #  test data
    weights = "./checkpoints/"
    best_models = "./checkpoints/bestmodels/"
    submit = "./submit/"
    model_name = "bninception_bcelog"
    num_classes = 28
    img_weight = 512
    img_height = 512
    channels = 4
    lr = 0.03
    batch_size = 32
    epochs = 60

config = DefaultConfigs()
