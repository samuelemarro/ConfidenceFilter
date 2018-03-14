import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import argparse
import tensorboard

#TODO:
##Inserire preprocessing?
##Implementare una rete ufficiale (ResNet, VGG)
##La performance del training con output è più bassa
##Controllare il funzionamento corretto
##Softmax

class CollectionNetwork(nn.Module):
    def __init__(self, filtered_modules):
        super().__init__()
        self.filtered_modules = filtered_modules

    def forward(self, x, index):
        return self.filtered_modules[index].forward(x)

    def partial_output(self, x, index):
        return self.filtered_modules[index].partial_output(x)

    def forward_modules(self, index):
        return self.filtered_modules[index].forward_modules()

    def partial_output_modules(self, index):
        return self.filtered_modules[index].partial_output_modules()

    def length(self):
        return len(self.filtered_modules)

    def set_requires_grad(self, index, requires_grad):
        pass

class ConfidenceFilter(nn.Module):
    def __init__(self, filtered_network, confidence_functions, pretrained=False):
        super(ConfidenceFilter, self).__init__()

        self.pretrained = pretrained
        self.filtered_network = filtered_network

        if(isinstance(confidence_functions, list)):
            self.confidence_functions = confidence_functions
            if len(confidence_functions) != filtered_network.length() - 1 and len(confidence_functions) != 1:
                raise ValueError('confidence_functions must either be a single function or a list with length '
                                 'filtered_network.length() - 1. Length of confidence_functions: {};'
                                 'filtered_network.length(): {}'.format(len(confidence_functions), filtered_network.length()))
        else:
            self.confidence_functions = [confidence_functions] * (self.filtered_network.length() - 1)
        
    def forward(self, input):
        current_input = input
        current_x = input

        output_list = []
        indices_list = []
        batch_size = input.size()[0]

        #tracking_indices is used to track which positions are yet to be stored
        tracking_indices = Variable(torch.arange(batch_size).cuda())

        for i in range(self.filtered_network.length()):
            current_x = self.filtered_network.forward(current_x, i)

            if i == self.filtered_network.length() - 1:
                #If we reached the last module, save directly without filtering
                output_list.append(current_x)
                indices_list.append(current_indices)
            else:
                #All modules but the last have a partial_output and a confidence_function
                #The confidence_function is used to evaluate which elements of the partial_output
                #are confident enough to be stored
                partial_output = self.filtered_network.partial_output(current_x, i)
                confidence_function = self.confidence_functions[i]

                confidence_mask = confidence_function(partial_output)

                #We store the confident elements and keep the unconfident ones
                confident_indices = torch.squeeze(torch.nonzero(confidence_mask))
                current_indices = torch.squeeze(torch.nonzero(confidence_mask == 0))

                if confident_indices.shape[0] != 0:
                    #If some elements are confident enough, we store them along with their position
                    #in the batch
                    filtered_output = torch.index_select(partial_output, 0, confident_indices)
                    filtered_indices = torch.index_select(tracking_indices, 0, confident_indices).long()

                    output_list.append(filtered_output)
                    indices_list.append(filtered_indices)

                if current_indices.shape[0] == 0:
                    #If there aren't any elements left, there's no point in running the other modules
                    break
                else:
                    if isinstance(current_x, tuple):
                        #If the network has multiple elements we filter each one separately
                        current_x = tuple([torch.index_select(element, 0, current_indices) for element in current_x])
                    else:
                        current_x = torch.index_select(current_x, 0, current_indices)

                    #Update tracking_indices to match current_x
                    tracking_indices = torch.index_select(tracking_indices, 0, current_indices)

        
        output_shape = [batch_size] + list(output_list[0].size()[1:])

        unsorted_output = torch.cat(output_list)
        unsorted_indices = torch.cat(indices_list)

        #Sort the output according to the indices
        sorted_output = Variable(torch.zeros(output_shape).cuda()).index_copy(0, unsorted_indices, unsorted_output)

        return sorted_output

    def get_output(self, input, index):
        x = input

        #Run the previous modules
        for i in range(index):
            x = self.filtered_network.forward(x, i)

        #Don't train the previous modules
        x = Variable(x.data)

        x = self.filtered_network.forward(x, index)

        if self.pretrained:
            #If the network is pretrained we must not update its weights
            x = Variable(x.data)

        if(index != self.filtered_network.length() - 1):
            #The output of the last step is already included in forward()
            #It also must be trained, so we do not detach it
            x = self.filtered_network.partial_output(x, index)
        return x

    def get_parameters(self, index):
        modules = []
        
        if not self.pretrained:
            modules += self.filtered_network.forward_modules(index)
        if index != self.filtered_network.length() - 1:
            modules += self.filtered_network.partial_output_modules(index)

        parameters = []
        for module in modules:
            parameters += module.parameters()

        return parameters


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print('')

class FilteredModule(nn.Module):
    def __init__(self, filter_x, filter_input):
        super().__init__()
        self.filter_x = filter_x
        self.filter_input = filter_input

    def hidden(self, filtered_x, filtered_input):
        pass
    def output(self, filtered_x):
        pass

    def forward(self, x):
        hidden_result = self.hidden(x)
        output_result = self.output(hidden_result)
        return output_result

class ConvNet(FilteredModule):
    def __init__(self, filter_x):
        super(ConvNet, self).__init__(filter_x, True)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5 + (100 if filter_x else 0), 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

    def hidden(self, previous_hidden, input):
        input = self.pool(F.relu(self.conv1(input)))
        input = self.pool(F.relu(self.conv2(input)))
        input = input.view(-1, 16 * 5 * 5)
        if self.filter_x:
            input = torch.cat([input, previous_hidden], 1)
        input = F.relu(self.fc1(input))
        input = F.relu(self.fc2(input))
        return input

    def output(self, x):
        return self.fc3(x)

class FeedForwardNet(nn.Module):
    def __init__(self, hidden_layers, input_units, hidden_units, output_units, use_output, hidden_activation=F.relu, output_activation=None):
        super().__init__()

        self.use_output = use_output
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.hidden_layers = nn.ModuleList()

        first_hidden_layer = nn.Linear(input_units, hidden_units)
        self.hidden_layers.append(first_hidden_layer)

        for i in range(hidden_layers - 1):
            hidden_layer = nn.Linear(hidden_units, hidden_units)
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(hidden_units, output_units)
        self.partial_output_layer = nn.Linear(hidden_units, output_units)

    def forward(self, x):
        previous_shape = list(x.shape)

        if previous_shape == []:
            return x

        x = x.view(x.shape[0], -1)

        for hidden_layer in self.hidden_layers:
            try:
                x = hidden_layer(x)
            except:
                print(x)
                raise
            if self.hidden_activation != None:
                x = self.hidden_activation(x)

        if self.use_output:
            x = self.output_layer(x)

            if self.output_activation != None:
                x = self.output_activation(x)
        return x

    def forward_modules(self):
        return list(self.hidden_layers) + [self.output_layer]

    def partial_output(self, x):
        x = self.partial_output_layer(x)

        if self.output_activation != None:
            x = self.output_activation(x)

        return x

    def partial_output_modules(self):
        return [self.partial_output_layer]

class Filter:
    def __init__(self, batch_size, use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()

        tracking_indices = torch.arange(batch_size).long()

        if self.use_cuda:
            tracking_indices = tracking_indices.cuda()

        self.tracking_indices = Variable(tracking_indices)

        self.current_indices = None
        self.confident_indices = None

        self.completed_registration = False

        self.filtered_outputs = []
        self.filtered_indices = []

    def filter(self, x):
        if self.completed_registration:
            #raise RuntimeError('Attempted to filter a tensor when all output '
            #                   'elements were registered. If you want to know '
            #                   'if all output elements are already registered, '
            #                   'use completed_registration or check the return '
            #                   'value of register_output.')
            x = torch.zeros([0] + list(x.shape[1:]))
            if self.use_cuda:
                x = x.cuda()
        else:
            x = torch.index_select(x, 0, self.current_indices)

        return x

    def register_output(self, output, confidence_function):
        if not self.completed_registration:
            if confidence_function == None:
                self.filtered_outputs.append(output)
                self.filtered_indices.append(self.tracking_indices)
                self.completed_registration = True
            else:
                confidence_mask = confidence_function(output)
                self.confident_indices = torch.squeeze(torch.nonzero(confidence_mask))
                self.current_indices = torch.squeeze(torch.nonzero(confidence_mask == 0))

                if self.confident_indices.shape[0] != 0:
                    filtered_output = torch.index_select(output, 0, self.confident_indices)
                    filtered_indices = torch.index_select(self.tracking_indices, 0, self.confident_indices)
                    print(filtered_indices)

                    self.filtered_outputs.append(filtered_output)
                    self.filtered_indices.append(filtered_indices)

                if self.current_indices.shape[0] == 0:
                    self.completed_registration = True
                else:
                    self.tracking_indices = torch.index_select(self.tracking_indices, 0, self.current_indices)

    def reordered_output(self):
        if not self.completed_registration:
            raise RuntimeError('Not all output elements were registered. If you want '
                               'to register all the output elements without filtering, '
                               'use register_output(output, None).')

        final_indices = torch.cat(self.filtered_indices)
        final_outputs = torch.cat(self.filtered_outputs)

        output_zeros = torch.zeros(final_outputs.shape)

        if self.use_cuda:
            output_zeros = output_zeros.cuda()

        return Variable(output_zeros).index_copy(0, final_indices, final_outputs)

class CustomResNet(models.ResNet):
    def __init__(self, block, layers, model_url, confidence_functions, num_classes = 1000):
        super(CustomResNet, self).__init__(block, layers, num_classes)
        self.confidence_functions = confidence_functions
        self.training = True
        self.load_state_dict(torch.utils.model_zoo.load_url(model_url))
        
        self.output_layers = []
    def forward(self, x):
        if training:
            return super().forward(x)
        else:
            filter = Filter(x.shape[0])

            for i in range(6):
                x = run_step(x, i)
                #output = pass
                filter.register_output(output, lambda x: cutoff_mask(x, 0.5))

                if filter.completed_registration:
                    break
                
                x = filter.filter(x)

            return filter.reordered_output()

    def run_step(self, x, step):
        if list(x.shape) == []:
            return x

        if step == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        elif step == 1:
            x = self.layer1(x)
        elif step == 2:
            x = self.layer2(x)
        elif step == 3:
            x = self.layer3(x)
        elif step == 4:
            x = self.layer4(x)
        elif step == 5:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            raise ValueError('Invalid step number')

        return x

def cutoff_mask(x, minimum):
    x, _ = torch.max(x, 1)
    x = x > minimum
    return x

class Foo(nn.Module):
    def __init__(self):
        self.net0 = FeedForwardNet(5, 32 * 32 * 3, 100, 10)
        self.net1 = FeedForwardNet(5, 100, 100, 10)
        self.net2 = FeedForwardNet(5, 100, 100, 10)

    def foo(x, training):
        if training:
            x = self.net0.hidden(x)
            x = self.net1.hidden(x)
            x = self.net2.hidden(x)

            x = self.net2.output(x)

            return x
        else:
            filter = Filter(x.shape[0])

            x = self.net0.hidden(x)
            output0 = self.net0.output(x)
            filter.register_output(output0, lambda x: cutoff_mask(x, 0.5))

            x = filter.filter(x)

            x = self.net1.hidden(x)
            output1 = self.net1.output(x)
            filter.register_output(output1, lambda x: cutoff_mask(x, 0.5))

            x = filter.filter(x)

            x = self.net2.hidden(x)
            output2 = self.net2.output(x)
            filter.register_output(output2, None)

            return filter.reordered_output()
    
def train(network, criterion, trainloader, train_epochs, print_progress=True):

    filtered_module_count = network.filtered_network.length()

    if network.pretrained:
        filtered_module_count -= 1

    total_runs = filtered_module_count * len(trainloader) * train_epochs

    for filtered_module_index in range(filtered_module_count):
        optimizer = optim.SGD(network.get_parameters(filtered_module_index), lr=1e-3)
        for epoch in range(train_epochs):  # loop over the dataset multiple times
            for i, data in enumerate(trainloader):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = network.get_output(inputs, filtered_module_index)

                loss = criterion(outputs, labels)
                loss.backward()

                optimizer.step()

                if print_progress:
                    global_run = filtered_module_index * train_epochs * len(trainloader) + epoch * len(trainloader) + i + 1
                    if global_run % (total_runs // 100) == 0:
                        print_progress_bar(global_run, total_runs, length=50)

def main():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=False, num_workers=2)#Temp

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(classes[labels[j]] for j in range(4)))

    modules = nn.ModuleList([
        FeedForwardNet(3, 32 * 32 * 3, 100, 10, False),
        FeedForwardNet(3, 100, 100, 10, True)
        ])

    confidence_functions = [
        lambda x: cutoff_mask(x, -1000)
        ]

    net = ConfidenceFilter(CollectionNetwork(modules), confidence_functions, pretrained=False)
    #net = Foo()
    net.cuda()

    #print('Is CUDA: {}'.format(next(net.parameters()).is_cuda))

    criterion = nn.CrossEntropyLoss()

    print_progress = True
    train_epochs = 1

    train(net, criterion, trainloader, 2)
    
    '''for filtered_module_index in range(net.filtered_network.length()):
        for epoch in range(train_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net.get_output(inputs, filtered_module_index)
                #outputs = foo(x, True)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if print_progress:
                    global_run = filtered_module_index * train_epochs * len(trainloader) + epoch * len(trainloader) + i + 1
                    if global_run % (total_runs // 100) == 0:
                        print_progress_bar(global_run, total_runs, length=50)'''

    print('Finished Training')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        #outputs = foo(Variable(images), False)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', nargs='?', const=True, type=bool, default=True, help='If True, uses a CUDA-enabled GPU (if available)')
    FLAGS, unparsed = parser.parse_known_args()

    main()
