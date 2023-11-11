import torch, os, json
from torchvision import datasets, transforms
import pandas as pd
import cv2
from PIL import ImageFile
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils')
from MODULE_utils import clean_up_folder, check_path, get_lambda, cal_SPASL_score
sys.path.append('../visualize')
from MODULE_radar_chart import draw_radar_chart_1_network_SPASL_v1

ImageFile.LOAD_TRUNCATED_IMAGES = True

#model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#model_name = 'resnet18'
#testdata_dir = ''
# test_set_name = SuperHard-10

# test_set_name: SuperHard-10 SuperHard-50 SuperHard-100 

def test_on_one_component(model, model_name, component_dir, component_name, result_mother_folder, progress_bar = False, batch_size = 1):
    clean_up_folder(component_dir)
    
    IW_dir = f'{result_mother_folder}/IW_table/{model_name}'  # rename to IW_dir

    all_classes = sorted(os.listdir(component_dir))

    # create empty folders for results

    for each_class in all_classes:
        check_path(f'{IW_dir}/{each_class}')

    # locate the break point and resume
    starting_idx = 0

    for each_class in all_classes:
        if os.path.exists(f'{IW_dir}/{each_class}/{model_name}_{each_class}_{component_name}.csv'):
            tmp_df = pd.read_csv(f'{IW_dir}/{each_class}/{model_name}_{each_class}_{component_name}.csv')
            starting_idx += len(tmp_df)
        else:
            break

    # special input modifications for the following models
    if model_name == 'ViTB16':
        data_transforms = transforms.Compose([
            transforms.Resize((384,384)),
            transforms.ToTensor()])
    elif model_name == 'ViTH14':
        data_transforms = transforms.Compose([
            transforms.Resize((518,518)),
            transforms.ToTensor()])
    elif model_name == 'ViTL16':
        data_transforms = transforms.Compose([
            transforms.Resize((512,512)),
            transforms.ToTensor()])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()])
    
    # load the test data
    testset = datasets.ImageFolder(component_dir, transform=data_transforms)
    total_test_num = len(testset)
    
    if starting_idx == 0:
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        all_images = testloader.sampler.data_source.imgs[starting_idx:]
    else:
        print(f'Resuming from idx = {starting_idx}')
        testloader0 = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                 shuffle=False, num_workers=2)
        all_images = testloader0.sampler.data_source.imgs[starting_idx:]
        del testloader0
    
        sub_set_idx = [i for i in range(starting_idx, total_test_num)]
        sub_test_set = torch.utils.data.Subset(testset, sub_set_idx)
        testloader = torch.utils.data.DataLoader(sub_test_set, batch_size=1,
                                                 shuffle=False, num_workers=2)

    model.eval()
    model = model.cuda()

    for i in tqdm(range(len(testloader)), disable=not progress_bar):
        
        img, lbl = testloader.dataset[i][0].to('cuda'), torch.tensor(testloader.dataset[i][1], dtype=torch.int16).to('cuda')
        img = img[None,:]    
        name_split = all_images[i][0].replace('\\n', '/n')
        class_name = name_split.split('/')[-2] 
        if not os.path.exists(f'{IW_dir}/{class_name}/{model_name}_{class_name}_{component_name}.csv'):
            tmp_df = pd.DataFrame(columns=['filename', 'p0', 'p1', 'p2', 'p3', 'p4', 'correct_idx', 'lambda'])
            tmp_df.to_csv(f'{IW_dir}/{class_name}/{model_name}_{class_name}_{component_name}.csv', index=False)
        
        cur_df = pd.read_csv(f'{IW_dir}/{class_name}/{model_name}_{class_name}_{component_name}.csv')

        img = img.cuda()

        probabilities = torch.nn.functional.softmax(model(img)[0], dim=0)
        probs, predict_ids = torch.topk(probabilities, 5)
        cur_entry = [name_split.split('/')[-1]] # add filename

        correct_idx = -1

        for ii in range(5):
            if predict_ids[ii] == lbl.item():
                correct_idx = ii
            cur_entry.append(round(probs[ii].item(), 5)) # add 5 probabilities
        # calculate lambda
        cur_lambda = get_lambda(cur_entry[1:])
        # append correct_idx and lambda
        cur_entry.append(correct_idx)
        cur_entry.append(cur_lambda)
        
        cur_df.loc[-1] = cur_entry
        cur_df.to_csv(f'{IW_dir}/{class_name}/{model_name}_{class_name}_{component_name}.csv', index=False)
    
    # calculate number of R1 predictions and wrong predictions
    # acc1 = R1 / total
    # acc5 = (total - wrong) / total
    R1_count = 0
    Wrong_count = 0
    for each_class in all_classes:
        cur_df = pd.read_csv(f'{IW_dir}/{each_class}/{model_name}_{each_class}_{component_name}.csv')
        R1_count += (cur_df['correct_idx'] == 0).sum()
        Wrong_count += (cur_df['correct_idx'] == -1).sum()
    acc1 = round(R1_count * 100 / total_test_num, 2)
    acc5 = round((total_test_num - Wrong_count) * 100 / total_test_num, 2)
    
    print(f'{model_name} on {component_name}: # R1 = {R1_count}, # Wrong = {Wrong_count}, total # = {total_test_num}. Top-1: {acc1}%, Top-5: {acc5}%')
    
    return acc1, acc5      

def test_on_whole_SPASL_benchmark(model, model_name, SPASL_dir, result_mother_folder, bm_name = 'SPASL_v1', draw_radar_graph = False, progress_bar = False, batch_size = 1, force_to_execute = False):
    """

        force_to_execute: default as "False". Sometimes there is a record in the history but IW-tables are missing. If image-wise information are important, you can force the execution of this function.
    """
    check_path(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}')
    if not force_to_execute:
        # check history csv, if not exist, proceed
        if os.path.exists(f'{result_mother_folder}/result_on_whole_SPASL/history.csv'):
            df0 = pd.read_csv(f'{result_mother_folder}/result_on_whole_SPASL/history.csv')
            if model_name in list(df0['model']):
                print(f'A record is found in the history:')
                print(df0.loc[df0['model']==model_name])
                if not os.path.exists(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}/{model_name}_radar_chart_on_{bm_name}.png'):
                    acc_1 = list(df0.loc[df0['model'] == model_name].iloc[-1, 1:6])
                    df0 = pd.read_csv(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}/{model_name}_top5_acc.csv')
                    acc_5 = list(df0.iloc[-1, 1:6])
                    draw_radar_chart_1_network_SPASL_v1(model_name, acc_1, acc_5, bm_name, save_path = f'{result_mother_folder}/result_on_whole_SPASL/{model_name}', show_plot= draw_radar_graph)
                    print(f'The radar chart is saved to {result_mother_folder}/result_on_whole_SPASL/{model_name}')
                else:
                    if draw_radar_graph:
                        img1 = cv2.imread(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}/{model_name}_radar_chart_on_{bm_name}.png')
                        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                        plt.imshow(img1)
                return
        else:
            df0 = pd.DataFrame(columns=['model', 'SH', 'PI', 'AA', 'SN', 'LR', 'score'])
            df0.to_csv(f'{result_mother_folder}/result_on_whole_SPASL/history.csv', index=False)
            # create an empty json file
            json_data = {}
            save_to_file = open(f'{result_mother_folder}/result_on_whole_SPASL/history.json', 'w')
            json.dump(json_data, save_to_file)
            save_to_file.close()

    
    components = os.listdir(SPASL_dir)
    assert len(components)==5, 'There are fewer than 5 components, if you are trying to test on specific component(s), please call test_on_one_component()'

    df1 = pd.read_csv(f'{result_mother_folder}/result_on_whole_SPASL/history.csv')
    
    cur_entry1 = [model_name]
    cur_entry5 = [model_name]
    for each_component in ['SH','PI','AA','SN','LR']:
        cur_acc1, cur_acc5 = test_on_one_component(model, model_name, f'{SPASL_dir}/{each_component}', each_component, result_mother_folder, progress_bar, batch_size)
        cur_entry1.append(cur_acc1)
        cur_entry5.append(cur_acc5)
        
        # write to the table
    score1 = cal_SPASL_score(cur_entry1[1:])
    score5 = cal_SPASL_score(cur_entry5[1:])
    cur_entry1.append(score1)
    cur_entry5.append(score5)
    
    df1.loc[-1] = cur_entry1
    df1.to_csv(f'{result_mother_folder}/result_on_whole_SPASL/history.csv', index=False)

    tmp_df = pd.DataFrame(columns=['model', 'SH', 'PI', 'AA', 'SN', 'LR', 'score'])
    tmp_df.loc[0] = cur_entry1
    tmp_df.to_csv(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}/{model_name}_top1_acc.csv', index=False)
    tmp_df.loc[0] = cur_entry5
    tmp_df.to_csv(f'{result_mother_folder}/result_on_whole_SPASL/{model_name}/{model_name}_top5_acc.csv', index=False)

    new_json_entry = {'acc1':cur_entry1[1:], 'acc5':cur_entry5[1:]}
    f0 = open(f'{result_mother_folder}/result_on_whole_SPASL/history.json', 'r')
    json_data = json.loads(f0.read())
    f0.close()
    json_data.update({cur_entry1[0]:new_json_entry})
    save_to_file = open(f'{result_mother_folder}/result_on_whole_SPASL/history.json', 'w')
    json.dump(json_data, save_to_file)
    save_to_file.close()
    print('*'*10)
    print(f'Model {model_name} SPASL score: {score1} and {score5}')
    print('*'*10)
    print(f'IW-tables are saved to {result_mother_folder}/IW_table/{model_name}')
    print(f'Summary (accuracies, scores) is saved to {result_mother_folder}/result_on_whole_SPASL/{model_name}')
    print(f'Summary is also added to test history at {result_mother_folder}/result_on_whole_SPASL/history.csv')
    if draw_radar_graph:        
        draw_radar_chart_1_network_SPASL_v1(model_name, cur_entry1[1:6], cur_entry5[1:6], 'SPASL_demo', save_path = f'{result_mother_folder}/result_on_whole_SPASL/{model_name}', show_plot= True)
        print(f'The radar chart is saved to {result_mother_folder}/result_on_whole_SPASL/{model_name}')
    
    
    