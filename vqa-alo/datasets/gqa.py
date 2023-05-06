import os, h5py, re, json
import torch
import numpy as np
from os import path as osp
from bootstrap.lib.logger import Logger
from bootstrap.datasets.dataset import Dataset
from bootstrap.datasets import transforms as bootstrap_tf
from tqdm import tqdm

class GQAOOD(Dataset):

    def __init__(self,
            dir_data='data/gqa/gqaood',
            split='train',
            batch_size=80,
            nb_threads=4,
            pin_memory=False,
            shuffle=False,
            nans=1000,
            minwcount=10,
            nlp='mcb',
            proc_split='train',
            dir_rcnn='data/gqa/gqaood/annotations/objects/',
            dir_cnn=None,
            ):
        super(GQAOOD, self).__init__(
            dir_data=dir_data,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            nb_threads=nb_threads
        )
        self.dir_raw = dir_data
        self.nans = nans
        self.minwcount = minwcount
        self.nlp = nlp
        self.subprocess_dir = osp.join(self.dir_data, 'processed')
        self.path_wid_to_word = osp.join(self.subprocess_dir, 'wid_to_word.pth')
        self.path_word_to_wid = osp.join(self.subprocess_dir, 'word_to_wid.pth')
        self.path_aid_to_ans = osp.join(self.subprocess_dir, 'aid_to_ans.pth')
        self.path_ans_to_aid = osp.join(self.subprocess_dir, 'ans_to_aid.pth')
        self.path_trainset = osp.join(self.subprocess_dir, 'trainset.pth')
        self.path_val_all_set = osp.join(self.subprocess_dir, 'val_all_set.pth')
        self.path_val_head_set = osp.join(self.subprocess_dir, 'val_head_set.pth')
        self.path_val_tail_set = osp.join(self.subprocess_dir, 'test_tail_set.pth')
        self.path_test_all_set = osp.join(self.subprocess_dir, 'test_all_set.pth')
        self.path_test_head_set = osp.join(self.subprocess_dir, 'test_head_set.pth')
        self.path_test_tail_set = osp.join(self.subprocess_dir, 'test_tail_set.pth')

        self.download()
        self.process()
        self.wid_to_word = torch.load(self.path_wid_to_word)
        self.word_to_wid = torch.load(self.path_word_to_wid)
        self.aid_to_ans = torch.load(self.path_aid_to_ans)
        self.ans_to_aid = torch.load(self.path_ans_to_aid)

        # here we use testdev split to be test set, because GQA-OOD does not include test-ood split.
        if split == 'train':
            self.dataset = torch.load(self.path_trainset)
        elif split == 'val_all':
            self.dataset = torch.load(self.path_val_all_set)
        elif split == 'val_head':
            self.dataset = torch.load(self.path_val_head_set)
        elif split == 'val_tail':
            self.dataset = torch.load(self.path_val_tail_set)
        elif split == 'testdev_all':
            self.dataset = torch.load(self.path_test_all_set)
        elif split == 'testdev_head':
            self.dataset = torch.load(self.path_test_head_set)
        else:
            self.dataset = torch.load(self.path_test_tail_set)

        self.dir_rcnn = dir_rcnn
        self.dir_cnn = dir_cnn

        self.collate_fn = bootstrap_tf.Compose([
            bootstrap_tf.ListDictsToDictLists(),
            bootstrap_tf.PadTensors(use_keys=[
                'question', 'pooled_feat', 'cls_scores', 'rois', 'cls', 'cls_oh', 'norm_rois'
            ]),
            # bootstrap_tf.SortByKey(key='lengths'), # no need for the current implementation
            bootstrap_tf.StackTensors()
        ])
    # this is only compatible for open ended VQA accuracy
    def get_subtype(self):
        return 'type'

    def process(self):
        if not osp.exists(self.subprocess_dir):
            os.mkdir(self.subprocess_dir)
            dir_questions = osp.join(self.dir_data, 'annotations', 'questions')
            train_question_dir = osp.join(dir_questions, 'train_balanced_questions.json')
            val_all_question_dir = osp.join(dir_questions, 'ood_val_all.json')
            val_head_question_dir = osp.join(dir_questions, 'ood_val_head.json')
            val_tail_question_dir = osp.join(dir_questions, 'ood_val_tail.json')
            testdev_all_question_dir = osp.join(dir_questions, 'ood_testdev_all.json')
            testdev_head_question_dir = osp.join(dir_questions, 'ood_testdev_head.json')
            testdev_tail_question_dir = osp.join(dir_questions, 'ood_testdev_tail.json')

            with open(file=train_question_dir, mode='r') as f:
                train_balanced = json.load(f)
            trainset = self.proc_ques(train_balanced)
            trainset = self.tokenize_questions(trainset, self.nlp)

            with open(file=val_all_question_dir, mode='r') as f:
                val_all = json.load(f)
            val_all_set = self.proc_ques(val_all)
            val_all_set = self.tokenize_questions(val_all_set, self.nlp)

            with open(file=val_head_question_dir, mode='r') as f:
                val_head = json.load(f)
            val_head_set = self.proc_ques(val_head)
            val_head_set = self.tokenize_questions(val_head_set, self.nlp)

            with open(file=val_tail_question_dir, mode='r') as f:
                val_tail = json.load(f)
            val_tail_set = self.proc_ques(val_tail)
            val_tail_set = self.tokenize_questions(val_tail_set, self.nlp)

            with open(file=testdev_all_question_dir, mode='r') as f:
                testdev_all = json.load(f)
            testdev_all_set = self.proc_ques(testdev_all)
            testdev_all_set = self.tokenize_questions(testdev_all_set, self.nlp)

            with open(file=testdev_head_question_dir, mode='r') as f:
                testdev_head = json.load(f)
            testdev_head_set = self.proc_ques(testdev_head)
            testdev_head_set = self.tokenize_questions(testdev_head_set, self.nlp)

            with open(file=testdev_tail_question_dir, mode='r') as f:
                testdev_tail = json.load(f)
            testdev_tail_set = self.proc_ques(testdev_tail)
            testdev_tail_set = self.tokenize_questions(testdev_tail_set, self.nlp)

            top_words, wcounts = self.top_words(trainset, val_all_set, self.minwcount)
            wid_to_word = {i + 1: w for i, w in enumerate(top_words)}
            word_to_wid = {w: i + 1 for i, w in enumerate(top_words)}

            trainset = self.insert_UNK_token(trainset, wcounts, self.minwcount)
            val_all_set = self.insert_UNK_token(val_all_set, wcounts, self.minwcount)
            val_head_set = self.insert_UNK_token(val_head_set, wcounts, self.minwcount)
            val_tail_set = self.insert_UNK_token(val_tail_set, wcounts, self.minwcount)
            testdev_all_set = self.insert_UNK_token(testdev_all_set, wcounts, self.minwcount)
            testdev_head_set = self.insert_UNK_token(testdev_head_set, wcounts, self.minwcount)
            testdev_tail_set = self.insert_UNK_token(testdev_tail_set, wcounts, self.minwcount)

            top_answers = self.top_answers(trainset, val_all_set, self.nans)
            aid_to_ans = [a for i, a in enumerate(top_answers)]
            ans_to_aid = {a: i for i, a in enumerate(top_answers)}
            trainset = self.encode_questions(trainset, word_to_wid, ans_to_aid)
            val_all_set = self.encode_questions(val_all_set, word_to_wid, ans_to_aid)
            val_head_set = self.encode_questions(val_head_set, word_to_wid, ans_to_aid)
            val_tail_set = self.encode_questions(val_tail_set, word_to_wid, ans_to_aid)
            testdev_all_set = self.encode_questions(testdev_all_set, word_to_wid, ans_to_aid)
            testdev_head_set = self.encode_questions(testdev_head_set, word_to_wid, ans_to_aid)
            testdev_tail_set = self.encode_questions(testdev_tail_set, word_to_wid, ans_to_aid)

            torch.save(wid_to_word, self.path_wid_to_word)
            torch.save(word_to_wid, self.path_word_to_wid)
            torch.save(aid_to_ans, self.path_aid_to_ans)
            torch.save(ans_to_aid, self.path_ans_to_aid)
            torch.save(trainset, self.path_trainset)
            torch.save(val_all_set, self.path_val_all_set)
            torch.save(val_head_set, self.path_val_head_set)
            torch.save(val_tail_set, self.path_val_tail_set)
            torch.save(testdev_all_set, self.path_test_all_set)
            torch.save(testdev_head_set, self.path_test_head_set)
            torch.save(testdev_tail_set, self.path_test_tail_set)

    def encode_questions(self, questions, word_to_wid, ans_to_aid):
        for item in questions:
            item['question_wids'] = [word_to_wid[w] for w in item['question_tokens_UNK']]
            item['answer_id'] = ans_to_aid.get(item['answer'], len(ans_to_aid) - 1)
        return questions

    def top_answers(self, annotations1, annotations2, nans):
        counts = {}
        for item in tqdm(annotations1):
            ans = item['answer']
            counts[ans] = counts.get(ans, 0) + 1
        for item in tqdm(annotations2):
            ans = item['answer']
            counts[ans] = counts.get(ans, 0) + 1

        cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
        Logger()('Top answer and their counts:')
        for i in range(20):
            Logger()(cw[i])

        vocab = []
        nans1 = min(nans, len(cw))
        for i in range(nans1):
            vocab.append(cw[i][1])
        Logger()('Number of answers left: {} / {}'.format(nans1, len(cw)))
        return vocab[:nans]

    def insert_UNK_token(self, questions, wcounts, minwcount):
        for item in questions:
            item['question_tokens_UNK'] = [w if wcounts.get(w,0) > minwcount else 'UNK' for w in item['question_tokens']]
        return questions

    def proc_ques(self, input: dict):
        set = []
        for item in input.items():
            tmp_ques = {}
            tmp_ques['question_id'] = item[0]
            tmp_ques['imageId'] = item[1]['imageId']
            tmp_ques['question'] = item[1]['question']
            tmp_ques['answer'] = item[1]['answer']
            set.append(tmp_ques)
        return set

    def top_words(self, questions1, questions2, minwcount):
        wcounts = {}
        for item in questions1:
            for w in item['question_tokens']:
                wcounts[w] = wcounts.get(w, 0) + 1
        for item in questions2:
            for w in item['question_tokens']:
                wcounts[w] = wcounts.get(w, 0) + 1
        cw = sorted([(count,w) for w, count in wcounts.items()], reverse=True)
        Logger()('Top words and their wcounts:')
        for i in range(20):
            Logger()(cw[i])

        total_words = sum(wcounts.values())
        Logger()('Total words: {}'.format(total_words))
        bad_words = [w for w in sorted(wcounts) if wcounts[w] <= minwcount]
        vocab = [w for w in sorted(wcounts) if wcounts[w] > minwcount]
        bad_count = sum([wcounts[w] for w in bad_words])
        Logger()('Number of bad words: {}/{} = {:.2f}'.format(len(bad_words), len(wcounts), len(bad_words)*100.0/len(wcounts)))
        Logger()('Number of words in vocab would be {}'.format(len(vocab)))
        Logger()('Number of UNKs: {}/{} = {:.2f}'.format(bad_count, total_words, bad_count*100.0/total_words))
        vocab.append('UNK')
        return vocab, wcounts

    def tokenize_questions(self, questions, nlp):
        Logger()('Tokenize questions')
        if nlp == 'nltk':
            from nltk.tokenize import word_tokenize
        for item in tqdm(questions):
            ques = item['question']
            if nlp == 'nltk':
                item['question_tokens'] = word_tokenize(str(ques).lower())
            elif nlp == 'mcb':
                item['question_tokens'] = self.tokenize_mcb(ques)
            else:
                item['question_tokens'] = self.tokenize(ques)
        return questions

    def tokenize(self, sentence):
        return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if
                i != '' and i != ' ' and i != '\n'];

    def tokenize_mcb(self, s):
        t_str = s.lower()
        for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;']:
            t_str = re.sub(i, '', t_str)
        for i in [r'\-', r'\/']:
            t_str = re.sub(i, ' ', t_str)
        q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
        q_list = list(filter(lambda x: len(x) > 0, q_list))
        return q_list

    def add_rcnn_to_item(self, item, imgID):
        path_rcnn = os.path.join(self.dir_rcnn, '{}.pth'.format(item['image_name']))
        item_rcnn = torch.load(path_rcnn)
        item['visual'] = item_rcnn['pooled_feat']
        item['coord'] = item_rcnn['rois']
        item['norm_coord'] = item_rcnn['norm_rois']
        item['nb_regions'] = item['visual'].size(0)
        return item

    def __getitem__(self, index):
        item = {}
        item['index'] = index

        # Process Question (word token)
        question = self.dataset[index]
        item['question_id'] = question['question_id']
        item['question'] = torch.LongTensor(question['question_wids'])
        item['lengths'] = torch.LongTensor([len(question['question_wids'])])
        item['imageID'] = question['imageId']
        item['answer_id'] = torch.tensor([question['answer_id']], dtype=torch.long)
        item['class_id'] = torch.tensor([question['answer_id']], dtype=torch.long)

        # Process Object, Attribut and Relational features
        item = self.load_img_feats(item)
        return item

    def __len__(self):
        return len(self.dataset)

    def load_img_feats(self, item):
        frcn_feat_path = osp.join(self.dir_data, 'annotations/objects/objectslist', item['imageID'] + '.npz')
        frcn_feat = np.load(frcn_feat_path)
        frcn_feat_iter = self.proc_img_feat(frcn_feat['x'], img_feat_pad_size=100)
        item['visual'] = torch.from_numpy(frcn_feat_iter)
        # grid_feat = np.load(self.iid_to_grid_feat_path[iid])
        # grid_feat_iter = grid_feat['x']

        bbox_feat_iter = self.proc_img_feat(
            self.proc_bbox_feat(
                frcn_feat['bbox'],
                (frcn_feat['height'], frcn_feat['width'])
            ),
            img_feat_pad_size=100
        )
        item['norm_coord'] = torch.from_numpy(bbox_feat_iter)
        item['nb_regions'] = 100

        return item

    def proc_bbox_feat(self, bbox, img_shape):
        bbox_feat = np.zeros((bbox.shape[0], 4), dtype=np.float32)

        bbox_feat[:, 0] = bbox[:, 0] / float(img_shape[1])
        bbox_feat[:, 1] = bbox[:, 1] / float(img_shape[0])
        bbox_feat[:, 2] = bbox[:, 2] / float(img_shape[1])
        bbox_feat[:, 3] = bbox[:, 3] / float(img_shape[0])

        return bbox_feat

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat

    def download(self):
        dir_ann = osp.join(self.dir_raw, 'annotations')
        if not osp.exists(dir_ann):
            os.system('mkdir -p ' + dir_ann)
            os.system('wget -t 0 -c https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip -P' + dir_ann)
            os.system('wget -t 0 -c https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -P' + dir_ann)
            os.system('wget -t 0 -c https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -P' + dir_ann)
            # os.system('wget -t 0 -c https://downloads.cs.stanford.edu/nlp/data/gqa/spatialFeatures.zip -P' + dir_ann)
            os.system('wget -t 0 -c https://downloads.cs.stanford.edu/nlp/data/gqa/objectFeatures.zip -P' + dir_ann)
            os.system(f'mkdir {dir_ann}/sceneGraphs')
            os.system(f'mkdir {dir_ann}/questions')
            os.system(f'mkdir {dir_ann}/images')
            # os.system(f'mkdir {dir_ann}/spatialFeatures')
            os.system(f'mkdir {dir_ann}/objectFeatures')
            os.system(f'unzip -d {dir_ann}/sceneGraphs {dir_ann}/sceneGraphs.zip')
            os.system(f'unzip -d {dir_ann}/questions {dir_ann}/questions1.2.zip')
            os.system(f'unzip -d {dir_ann}/images {dir_ann}/images.zip ')
            # os.system(f'unzip -d {dir_ann}/images {dir_ann}/images.zip ')
            os.system(f'unzip -d {dir_ann}/images {dir_ann}/objectFeatures.zip')

            # download ood question splits
            os.system('wget -t -0 -c https://github.com/gqa-ood/GQA-OOD/tree/master/data' +
                      osp.join(dir_ann, 'questions'))


