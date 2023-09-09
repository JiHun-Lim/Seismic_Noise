import argparse
from config import config
from bbgan import bbgan

def train_generate(data_path:str, design:str, loss:str, test:int):
    
    train_config = config(data_path, design, loss, test)
    model = bbgan(train_config)

    model.train()
    model.generate(model.best_model_path, after_train = True)

def generate(data_path:str, design:str, loss:str, test:int):
    
    train_config = config(data_path, design, loss, test)
    model = bbgan(train_config)

    model.generate("model", after_train = False)

def train(data_path:str, design:str, loss:str, test:int):
    
    train_config = config(data_path, design, loss, test)
    model = bbgan(train_config)

    model.train()

def generate_h5py(data_path:str, design:str, loss:str, test:int):
    
    train_config = config(data_path, design, loss, test)
    model = bbgan(train_config)

    model.generate_h5py("model")

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type= str, default= 'data')
    parser.add_argument('--design', type= str, default= 'time')
    parser.add_argument('--loss', type=str, default='KL')
    args = parser.parse_args()

    #train(args.data_path, args.design, 'KL', 6)

    # train_generate(args.data_path, args.design, args.loss, 6)

    # train_generate(args.data_path, args.design, 'JS')

    generate(args.data_path, args.design, 'KL', 6)

    # generate_h5py(args.data_path, args.design, 'KL', 6)





