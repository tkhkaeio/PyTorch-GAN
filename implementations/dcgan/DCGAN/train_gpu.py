import torch
import torch.optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import os
import argparse
from network32x32 import *
from utils import *
from distutils.dir_util import copy_tree
from tensorboardX import SummaryWriter

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="DCGAN_IS_cifar10_test",type=str,
                        dest='name', help='to name the model')
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--save_colab_dir', default=None, type=str,
                        dest='save_colab_dir', help='the path of save generate images in google colaboratory')
    parser.add_argument('--load_model', default=False,type=bool,
                        dest='load_model', help='whether to load model')
    parser.add_argument('--restart', default=0, type=int,
                        dest='restart', help='the num of epoch to retrain')

    return parser.parse_args()

def construct_model(args, config):
    os.makedirs("./model", exist_ok=True)
    try:
        os.makedirs(args.save_colab_dir, exist_ok=True)
        os.makedirs(args.save_colab_dir + "/model", exist_ok=True)
    except:
        pass

    G = generator(z_size=config.z_size, out_size=config.channel_size, ngf=config.ngf).cuda()
    print('G network structure\n', G)
    print("G params: ", count_params(G))

    D = discriminator(in_size=config.channel_size, ndf=config.ndf).cuda()
    print('D network structure\n', D)
    print("D params: ", count_params(D))
    
    try:
        if args.load_model:
            load_folder_file = ("%s"%("/model"),
                                    "G_epoch_%d.pth.tar"%args.restart,
                                    "D_epoch_%d.pth.tar"%args.restart)
            load_checkpoint(G, load_folder_file[0], load_folder_file[1])
            load_checkpoint(D, load_folder_file[0], load_folder_file[2])
    except:
        print("Fail to load a model")       
    return G, D

def train_net(G, D, args, config):

    cudnn.benchmark = True
    traindir = args.train_dir
    cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    

    if config.dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(traindir, True,
                    transforms.Compose([transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]), download=True),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    elif config.dataset == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(traindir, train=True,
                     transform=transforms.Compose([transforms.Resize(config.image_size),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ]), download=True), 
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    elif config.dataset == 'celebA':
        train_loader = torch.utils.data.DataLoader(
                MydataFolder(traindir,
                    transform=transforms.Compose([transforms.Resize(config.image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    
    elif config.dataset == 'wikiart':
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir,
                    transform=transforms.Compose([
                        transforms.Resize((config.image_size, config.image_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ])),
                batch_size=config.batch_size, shuffle=True,
                num_workers=config.workers, pin_memory=True)
    else:
        return

    # setup loss function
    criterion = nn.BCELoss().cuda()

    # setup optimizer
    optimizerD = torch.optim.Adam(D.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=config.base_lr, betas=(config.beta1, 0.999))

    # setup some varibles
    batch_time = AverageMeter()
    data_time = AverageMeter()
    D_losses = AverageMeter()
    G_losses = AverageMeter()
    

    fixed_noise = torch.FloatTensor(8 * 8, config.z_size, 1, 1).normal_(0, 1)
    with torch.no_grad():
        fixed_noise = Variable(fixed_noise.cuda())

    end = time.time()

    D.train()
    G.train()
    D_loss_list = []
    G_loss_list = []
    IS_list = []
    FID_list = []
    #make dirs
    try:
        os.makedirs(args.save_colab_dir + "/result", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/result/image", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/result/IS", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/result/FID", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/log", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/log/loss", exist_ok=True)
        os.makedirs(args.save_colab_dir + "/log/progress", exist_ok=True)
    except:
        pass
    os.makedirs("./result", exist_ok=True)
    os.makedirs("./result/image", exist_ok=True)
    os.makedirs("./result/IS", exist_ok=True)
    os.makedirs("./result/FID", exist_ok=True)
    os.makedirs("./log", exist_ok=True)
    os.makedirs("./log/loss", exist_ok=True)
    writer = SummaryWriter(comment=args.name)
    for epoch in range(args.restart, config.epoches+args.restart):
        for i, (input, _) in enumerate(train_loader):
            '''
                Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            '''
            data_time.update(time.time() - end)

            batch_size = input.size(0)
            input_var = Variable(input.cuda())

            # Train discriminator with real data
            label_real = torch.ones(batch_size)
            label_real_var = Variable(label_real.cuda())

            D_real_result = D(input_var).squeeze()
            D_real_loss = criterion(D_real_result, label_real_var)

            # Train discriminator with fake data
            label_fake = torch.zeros(batch_size)
            label_fake_var = Variable(label_fake.cuda())

            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            D_fake_loss = criterion(D_fake_result, label_fake_var)

            # Back propagation
            D_train_loss = D_real_loss + D_fake_loss
            D_losses.update(D_train_loss.item())

            D.zero_grad()
            D_train_loss.backward()
            optimizerD.step()

            '''
                Update G network: maximize log(D(G(z)))
            '''
            noise = torch.randn((batch_size, config.z_size)).view(-1, config.z_size, 1, 1)
            noise_var = Variable(noise.cuda())
            G_result = G(noise_var)

            D_fake_result = D(G_result).squeeze()
            G_train_loss = criterion(D_fake_result, label_real_var)
            G_losses.update(G_train_loss.item())

            # Back propagation
            D.zero_grad()
            G.zero_grad()
            G_train_loss.backward()
            optimizerG.step()

            batch_time.update(time.time() - end)
            end = time.time()
            

            if (i + 1) % config.display == 0:
                print_log(epoch + 1, args.restart, config.epoches, i + 1, len(train_loader), config.base_lr,
                            config.display, batch_time, data_time, D_losses, G_losses)
                batch_time.reset()
                data_time.reset()
            elif (i + 1) == len(train_loader):
                print_log(epoch + 1,args.restart, config.epoches, i + 1, len(train_loader), config.base_lr,
                            (i + 1) % config.display, batch_time, data_time, D_losses, G_losses)
                batch_time.reset()
                data_time.reset()

        D_loss_list.append(D_losses.avg)
        G_loss_list.append(G_losses.avg)
        writer.add_scalar('log/loss/D_loss', D_losses.avg, epoch)
        writer.add_scalar('log/loss/G_loss', G_losses.avg, epoch)
        D_losses.reset()
        G_losses.reset()
        
        # plt the generate images 
        plot_result(args.name, G, fixed_noise, config.image_size, epoch + 1, "./result/image", writer,  is_gray=(config.channel_size == 1))

        
        # calculate inception score
        #IS = calc_IS(args.name, G, epoch + 1, "./result/IS", writer)
        IS, FID = calc_eval(args.name, G, epoch + 1,"./result/IS", "./result/FID", writer)
        #print(IS)
        writer.add_scalar("IS", np.array(IS), epoch+1)
        writer.add_scalar("FID", np.array(FID), epoch+1)
        IS_list.append(IS)
        FID_list.append(FID)

        # save the D and G and loss curve
        if(epoch%10 == 0):
            save_checkpoint({'epoch': epoch, 'state_dict': D.state_dict(),}, os.path.join("./model", 'D_epoch_{}'.format(epoch)))
            save_checkpoint({'epoch': epoch, 'state_dict': G.state_dict(),}, os.path.join("./model", 'G_epoch_{}'.format(epoch)))
            plot_loss(args.name, D_loss_list, G_loss_list, epoch + 1, args.restart, config.epoches, "./log/loss")
            plot_IS(args.name, IS_list, epoch + 1, args.restart, config.epoches, "./result/IS")
            plot_FID(args.name, FID_list, epoch + 1, args.restart, config.epoches, "./result/FID")

    create_gif(args.name, 1, args.restart, config.epoches, "./result/image")
    plot_loss(args.name, D_loss_list, G_loss_list, epoch + 1, args.restart, config.epoches, "./log/loss")
    plot_IS(args.name, IS_list, epoch + 1, args.restart, config.epoches, "./result/IS")
    plot_FID(args.name, FID_list, epoch + 1, args.restart, config.epoches, "./result/FID")
    try:
        copy_tree("./log/", args.save_colab_dir+"/log")
        copy_tree("./model/", args.save_colab_dir+"/model")
        copy_tree("./result/", args.save_colab_dir+"/result")
    except:
        print("Fail to copy log to colab dir")
    
    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./result/all_scalars.json")
    writer.close()

if __name__ == '__main__':

    #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    import os
    args = parse()
    config = Config(args.config)
    os.mkdir(args.name, exist_ok=True)
    os.chdir(args.name, exist_ok=True)
    G, D = construct_model(args, config)
    train_net(G, D, args, config)
