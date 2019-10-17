import sys
import os
sys.path.append('D:/CodeMonkey/WORKSPACE/SRTP-PROJECT/SEU-SRTP-PROJECT/my-model/srcnn')
import dataset as KITTI
import model as srcnn
import global_path as GPATH
import config as SConfig

def train(model,data_path,config):
    """Train the model."""
    # Training dataset.

    dataset_train = KITTI.KITTIDataset()
    dataset_train.load_kitti(data_path, "train")

    # Validation dataset
    dataset_val =  KITTI.KITTIDataset()
    dataset_val.load_kitti(data_path, "val")

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=2,
                layers='heads')



#train
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train SRCNN to detect KITTI.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file ")
    parser.add_argument('--target', required=False,
                        metavar="/path/to/target",
                        help="Path to detect target file ")
    args = parser.parse_args()
    # 验证参数
    if args.command =='detect':
        assert args.target
    # print path info
    print('before'+args.command)
    print('BASE_DIR--',GPATH.BASE_DIR)
    print('KITTI_DIR--', GPATH.KITTI_DIR)
    print('MODEL_DIR--', GPATH.KITTI_DIR)
    print('WEIGHT--', args.weights)
    print('LOG_DIR--', GPATH.LOG_DIR)

    # config
    if args.command =='train':
        config =SConfig.Config()
    config.display()

    # creat model
    # Create model
    if args.command == "train":
        model =  srcnn.SRCNN('training',GPATH.LOG_DIR,config)

    # Select weights file to load

    if args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "coco":
        weights_path = GPATH.COCOPATH
    else:
        weights_path = args.weights

    # Load weights

    if args.weights.lower() !='new':
        print("Loading weights ", weights_path)

        model.load_weights(weights_path, by_name=True)
    # Train or evaluate
    if args.command == "train":
        train(model, GPATH.KITTI_DIR, config)
