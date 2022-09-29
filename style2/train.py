import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import model
import utils
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nst = model.GetStyleModel
image_loader = utils.loader

def main(args):
    content_img_path = args.content_img_path
    style_img_path = args.style_img_path
    size = args.size
    steps = args.steps
    content_weight = args.c_weight
    style_weight = args.s_weight

    content_img, style_img = image_loader(content_img_path, style_img_path, size = size)

    input_img = content_img.clone().to(device)

    model, style_losses, content_losses  = nst(style_img, content_img)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img.requires_grad_(True)])

    step = [0]
    print('Optimizing..')
    run = [0]
    while run[0] <= steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img_path', type=str, default = 'images/dancing.jpg')
    parser.add_argument('--style_img_path', type=str, default = 'images/picasso.jpg')
    parser.add_argument('--size', type=int, default = 512, help='size of image')
    parser.add_argument('--steps', type=int, default = 300)
    parser.add_argument('--c_weight', type=int, default = 1, help='weighting of content')
    parser.add_argument('--s_weight', type=int, default = 100000, help='weighting of style')

    args = parser.parse_args()

    print(args)
    output = main(args)
    torchvision.utils.save_image(output, '/content/drive/MyDrive/style_transfer/results/result2.png')
