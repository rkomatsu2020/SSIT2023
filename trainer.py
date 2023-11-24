import argparse
from SSIT.train import TrainSSIT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input train info (target directories)")
    # Environment
    parser.add_argument("-gpu", "--gpu_no", default=0, type=int, help="Set gpu no")
    parser.add_argument("-epoch", "--n_epoch", default=50, type=int, help="Set N epoch")
    parser.add_argument("-bn", "--batch_size", default=5, type=int, help="Set Batch size")
    parser.add_argument("-interval", "--save_interval", default=5, type=int, help="Set the epoch of save interval")
    parser.add_argument("-resume", "--resume_epoch", default=None, type=int, help="Set the epoch to restart")
    # Dataset
    parser.add_argument("-src", "--src_dir", default="photo2art", type=str, help="Write the directory name under '@Dataset'")
    parser.add_argument("-train_dirA", "--trainA", default="domainA", type=str, help="Write the directory name under '@Dataset/src'")
    parser.add_argument("-train_dirB", "--trainB", default="domainB", type=str, help="Write the directory name under '@Dataset/src'")
    parser.add_argument("-test_dirA", "--testA", default="domainA", type=str, help="Write the directory name under '@Dataset/src'")
    parser.add_argument("-test_dirB", "--testB", default="domainB", type=str, help="Write the directory name under '@Dataset/src'")
    
    args = parser.parse_args()
    kargs = vars(args)
    # Train
    train = TrainSSIT(gpu_no=kargs["gpu_no"], 
                      n_epoch=kargs["n_epoch"], batch_size=kargs["batch_size"],
                      dataset_name=kargs["src_dir"], dataset_op=kargs)
    train(save_interval=kargs["save_interval"], resume_num=kargs["resume_epoch"])