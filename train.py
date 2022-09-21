import os
import sys
import time
import random
import string
import argparse
from os.path import join as opj

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# white space 추가했다
character_str = "!(),.00123456789? ㅁㅇㅈㅎ가각간갇갈감갑값갓갔강갖같갚갛개객갤걀걔거걱건걷걸검겁것겅겉게겐겔겠겨격겪견결겹겼경곁계고곡곤곧골곰곱곳공과곽관광괘괜괴굉교구국군굳굴굵굶굽굿궁궈권귀귓규귝균귤그극근글긁금급긋긍기긴긷길김깃깅깊까깍깎깐깔깜깝깡깥깨꺼꺾껌껍껏껑께껴꼬꼭꼰꼴꼼꼽꽁꽂꽃꽉꽤꾸꾼꿀꿈꿔뀌뀐끄끈끊끌끓끔끗끝끼낌나낙낚난날낡남납낫낭낮낯낱낳내낵낸냄냇냈냉냐냠냥너넉넌널넓넘넙넝넣네넥넬넷녀녁년념녕노녹논놀놈농높놓놔뇌뇨뇽누눈눔눕뉘뉴늄느늑는늘늙능늦늬니닉닌닐님닙닛닝다닥닦단닫달닭닮담답닷당닿대댁댄댐댓댕더덕던덜덟덤덥덧덩덮데덴델뎀뎃뎅도독돈돌돔돕돗동돼되된될됩두둑둔둘둠둡둥뒤뒷듀드득든듣들듦듬듭듯등디딕딘딜딥딧딩딪따딱딴딸땀땅때땜땡떙떠떡떤떨떳떻떼또똑똘뚜뚝뚫뚱뛰뜨뜩뜯뜰뜸뜻띄띠라락란랄람랍랐랑랗래랙랜램랩랫랬랭략량러럭런럴럼럽럿렁렇레렉렌렐렘렙렛려력련렬렵령례로록론롤롬롭롯롱뢰료룡루룩룸룹룻뤄류륙륜률륨륭르른를름릇릉릎리릭린릴림립릿링마막만많말맑맘맙맛망맞맡맣매맥맨맴맵맹맺머먹먼멀멈멋멍멎메멘멜멧멩며면멸명몇모목몬몰몸몹못몽묘무묵묶문묻물뭄뭇뭉뭐뭔뭘뭣뮈뮤므미믹민믿밀밉밌밍및밑바박밖반받발밝밟밤밥밧방밭배백밴밸뱀뱃뱅뱉버번벌범법벗벚베벡벤벧벨벼벽변별볍병볕보복볶본볼봄봇봉봐봤뵈뵙부북분불붉붐붓붕붙뷔뷰브븐블비빅빈빌빔빗빙빚빛빠빡빨빵빼뺏뺨뻐뻔뻗뻘뻬뼈뼉뽀뽁뽈뽑뽕뿌뿐쁘쁜쁠쁨삘사삭산살삶삼삽삿상새색샌샐샘생샤샨샬샵샷샹샾서석섞선설섬섭섯성세섹센셀셈셉셋셔션셜셨셰셸소속손솔솜솝솟송솥쇄쇠쇼숍수숙순숟술숨숫숭숯숲숴쉐쉬쉰쉴쉼쉽쉿슈슉슐스슨슬슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌈쌍쌓쌤써썩썬썰썸썹쎄쎈쎼쏘쏟쏠쏭쑝쑤쓰쓴쓸씀씌씨씩씬씹씻씽아악안앉않알앓암압앗았앙앞애액앤앨앰앵야약얀얄얇양얕얗얘어억언얹얻얼엄업없엇었엉엊엌엎에엑엔엘엠엣여역엮연열엷염엽엿였영옆예옙옛오옥온올옮옳옴옵옷옹옻와왁완왓왔왕왜왠외왼요욕용우욱운울움웃웅워원월웠웨웬웰웹위윈윌윗윙유육윤율융으윽은을음응의이익인일읽잃임입잇있잉잊잎자작잔잖잘잠잡잣장잦재잭잼쟁쟈쟤저적전절젊점접젓정젖제젝젠젤젬젯져젼졌조족존졸좀좁종좋좌죄죠주죽준줄줌줍중줘쥐쥬즈즉즌즐즘즙증지직진짇질짐집짓징짙짚짜짝짧짬짱째쨌쩌쩍쩐쩔쩜쩡쪽쫄쫌쫓쭈쭉쭌쯤찌찍찜찢차착찬찮찰참찹찻창찾채책챌챔챙처척천철첨첩첫청체첸쳐초촉촌촛총촬최쵸추축춘출춤춥춧충취츄츠측츰층치칙친칠칡침칫칭카칵칸칼캄캇캉캐캔캘캠캡캣캬커컨컬컴컵컷케켄켈켐켑켓켜코콕콘콜콤콩쾌쿄쿠쿡쿤쿨쿱퀘퀴퀵퀸퀼큐큘크큰클큼키킥킨킬킴킹타탁탄탈탉탐탑탓탕태택탠탭터턱턴털텀텅테텍텐텔템토톡톤톨톰톱통퇴투툰툴툼퉁튀튜튤튬트특튼튿틀틈티틱틴틸팀팅파팍팎판팔팜팝팡팥패팩팬팰팽퍼퍽펀펌펍페펙펜펠펫펴편펼평폐포폭폰폴폼표푸푹풀품풋풍퓨프픈플픔피픽핀필핌핍핏핑하학한할함합핫항해핵핸햄햇했행향햬허헌험헛헝헤헥헨헬헸혀혁현혈혐협형혜호혹혼홀홈홉홍화확환활황회획횟횡효후훈훌훔훙훠훨휘휴휼흉흐흑흔흘흙흠흡흥흩희흰히힌힐힘"
def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the images containing characters which are not in opt.character')
        print('Filtering the images whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(opj(opt.save_dir, f'saved_models/{opt.exp_name}/log_dataset.txt'), 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.val_batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from warpctc_pytorch import CTCLoss 
            criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
    
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    
    
    if opt.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=opt.T_max, eta_min=opt.min_lr)
    elif opt.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=opt.T_0, eta_min=opt.min_lr)
    elif opt.scheudler == "":
        scheduler = None

    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(opj(opt.save_dir, f'saved_models/{opt.exp_name}/opt.txt'), 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        loss_avg.add(cost)

        # validation part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see training progress, we also conduct validation when 'iteration == 0' 
            elapsed_time = time.time() - start_time
            # for log
            with open(opj(opt.save_dir, f'saved_models/{opt.exp_name}/log_train.txt'), 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # training loss and validation loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), opj(opt.save_dir, f'saved_models/{opt.exp_name}/best_accuracy.pth'))
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), opj(opt.save_dir, f'saved_models/{opt.exp_name}/best_norm_ED.pth'))
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), opj(opt.save_dir, f'saved_models/{opt.exp_name}/iter_{iteration+1}.pth')
                )

        if (iteration + 1) == opt.num_iter:
            print('end the training')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4) 
    parser.add_argument('--tr_batch_size', type=int, default=32, help='train mode, input batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help="valid mode, input batch size")

    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--optimizer', default='adamw', help="which optimizer used")
    parser.add_argument('--scheduler', default='', help='what scheudler used')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument("--min_lr", type=float, default=1e-6, help="min lr")
    parser.add_argument("--T_max", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default =1e-6)
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    
    # modify
    
    parser.add_argument('--select_data', type=str, default='/',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                            help='assign ratio for each selected data in the batch')
    parser.add_argument('--character', type=str,default=character_str, help='character label')


    """ Data processing """
    # parser.add_argument('--select_data', type=str, default='MJ-ST',
    #                     help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    # parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
    #                     help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # parser.add_argument('--character', type=str,
    #                     default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    # Modify
    parser.add_argument('--save_dir', default="./saved_models/", help='path to save ouptut file')
    parser.add_argument("--is_kaggle", default=False, help="for package dependency")
    opt = parser.parse_args()

    if not hasattr(opt, "batch_size"):
        opt.batch_size = opt.tr_batch_size
        
    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    #os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)
    
    #os.makedirs(opj(opt.save_dir,f'/saved_models/{opt.exp_name}'), exist_ok=True)
    os.makedirs(opt.save_dir+f'saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    if opt.is_kaggle:
        os.system('pip install natsort')
    train(opt)