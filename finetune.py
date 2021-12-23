'''
Runs all finetuning tasks
'''
import subprocess
import os
import consts
import argparse


class Job:
    def __init__(self, task: str, tokenizer: str, ckpt: str, seed: int,
                 debug=False, two_level_embeddings=False, use_base=False,
                 use_long=False, use_shuffled=False, use_sp=False, use_cws=False,
                 use_no_index=False, use_byte=False, use_random_index=False,
                 classification_split_char=False, noise_type=None, 
                 noise_train=None, noise_test=None, fewshot=False):
        self.task = task
        self.tokenizer = tokenizer
        self.ckpt = ckpt
        self.seed = seed
        self.debug = debug
        self.fewshot = fewshot
        self.two_level_embeddings = two_level_embeddings
        self.use_base = use_base
        self.use_long = use_long
        self.use_shuffled = use_shuffled
        self.use_sp = use_sp
        self.use_no_index = use_no_index
        self.use_cws = use_cws
        self.use_byte = use_byte
        self.use_random_index = use_random_index
        self.classification_split_char = classification_split_char
        self.noise_type = noise_type
        self.noise_train = noise_train
        self.noise_test = noise_test
        
        # Generate other variables
        self.tokenizer_type = self.get_tokenizer_type()
        self.vocab_file = self.get_vocab_file()
        self.vocab_model_file = self.vocab_file.replace('.vocab', '.model')
        if self.use_cws:
            self.cws_vocab_file = self.vocab_file.replace('.vocab', '.cws_vocab')
        self.train_dir, self.dev_dir, self.test_dir = self.get_data_dirs()
        self.data_dir = self.test_dir
        self.config_file = self.get_config_file()
        self.dir_ckpts = self.get_dir_ckpts()
        self.init_checkpoint = os.path.join(self.dir_ckpts, ckpt + '.pt')
        self.output_dir = self.get_output_dir()
        if self.noise_type is not None:
            self.test_model = self.get_test_model()
        self.mode = self.get_mode()
        self.script = self.get_script()

    def get_tokenizer_type(self):
        if self.tokenizer == 'pinyin_concat_wubi':
            return 'PinyinConcatWubi'
        elif self.use_no_index:
            return 'CommonZhNoIndex'
        elif self.use_shuffled:
            return 'Shuffled'
        elif self.use_cws:
            return 'CWS'
        elif self.use_byte:
            return 'Byte'
        elif self.use_random_index:
            return 'RandomIndex'
        else:
            return consts.TOKENIZER_TYPES[self.tokenizer]

    def get_vocab_file(self):
        if self.use_no_index:
            return consts.VOCAB_FILES_NO_INDEX[self.tokenizer]
        if self.use_shuffled:
            return consts.VOCAB_FILES_SHUFFLED[self.tokenizer]
        if self.use_cws:
            return consts.VOCAB_FILES_CWS[self.tokenizer].format('80')
        if self.use_byte:
            return consts.VOC
        return consts.VOCAB_FILES[self.tokenizer]

    def is_classification_task(self) -> bool:
        C_TASKS = [
            'tnews'
            'iflytek',
            'wsc',
            'afqmc',
            'csl',
            'ocnli',
            'bq',
            'lcqmc',
            'thucnews',
        ]
        return self.task in C_TASKS

    def get_data_dirs(self):
        '''
        Return train_dir, dev_dir, test_dir
        '''
        def paths_append(paths, suf):
            for i in range(len(paths)):
                paths[i] = os.path.join(paths[i], suf)
            return paths

        dirs = [''] * 3
        dirs = paths_append(dirs, 'datasets')
        dirs = paths_append(dirs, self.task)

        if self.noise_type != None:
            # Use noisy data
            if self.noise_train > 0:
                raise NotImplementedError
            if self.noise_test > 0:
                dirs[2] = os.path.join(
                    dirs[2], 
                    'noisy', 
                    self.noise_type + '_' + str(self.noise_test))
            else:
                if self.fewshot:
                    raise NotImplementedError
                else:
                    # Some tasks doesn't have split directories
                    if self.task not in ['lcqmd', 'bq', 'thucnews']:
                        dirs[2] = os.path.join(dirs[2], 'split')
        else:
            # Not noisy
            # Handle fewshot and split dir
            if self.fewshot:
                dirs = paths_append(dirs, 'fewshot')
            else:
                # Some tasks doesn't have split directories
                if self.task not in ['lcqmd', 'bq', 'thucnews']:
                    dirs = paths_append(dirs, 'split')
        return tuple(dirs)

    def get_config_file(self):
        if self.use_base:
            return os.path.join('config', 'bert_base_config.json')
        else:
            return os.path.join('configs', 'bert_config_vocab22675.json')

    def get_dir_ckpts(self):
        if self.use_sp:
            return consts.DIR_CKPT_SP[self.tokenizer]
        elif self.use_long:
            return consts.DIR_CKPTS_LONG[self.tokenizer]
        elif self.use_no_index:
            return consts.DIR_CKPTS_NO_INDEX[self.tokenizer]
        elif self.use_shuffled:
            return consts.DIR_CKPTS_SHUFFLED[self.tokenizer]
        elif self.use_cws:
            return consts.DIR_CKPTS_CWS[self.tokenizer]
        else:
            return consts.DIR_CKPTS[self.tokenizer]

    def get_ckpt(self):
        if self.use_sp:
            return 'ckpt_8601'
        elif self.use_base:
            raise NotImplementedError
        elif self.use_no_index:
            return consts.BEST_CKPTS_NO_INDEX[self.tokenizer]
        elif self.tokenizer == 'pinyin_concat_wubi':
            raise NotImplementedError
        else:
            return consts.BEST_CKPTS[self.tokenizer]

    def get_epochs(self):
        if self.task == 'wsc':
            return 24
        if self.task == 'thucnews':
            return 4
        if self.task == 'cluener':
            return 12
        else:
            return 6

    def get_output_dir(self):
        task = self.task
        tokenizer = self.tokenizer
        if self.fewshot:
            task += '_fewshot'
        
        if self.use_sp:
            output_dir = os.path.join('logs', task, 'sp', tokenizer)
        else:
            if self.noise_type != None:
                task += '_{}_{}_{}'.format(self.noise_type,
                                           self.noise_train,
                                           self.noise_test)
            
            if self.is_classification_task() and self.classification_split_char:
                raise NotImplementedError
            
            if self.use_base:
                tokenizer += '_base'
            if self.use_long:
                tokenizer += '_long'
            if self.use_shuffled:
                tokenizer += '_shuffled'
            if self.use_no_index:
                tokenizer += '_no_index'
            if self.two_level_embeddings:
                tokenizer += '_twolevel'
            if self.use_cws:
                tokenizer += '_cws'

            if task == 'drcd':
                tokenizer += '_trad'  # DRCD always use traditional Chinese

            output_dir = os.path.join('logs', task, tokenizer)
        output_dir = os.path.join(output_dir, self.ckpt)
        return output_dir

    def get_test_model(self):
        noise_task = '_{}_{}_{}'.format(self.noise_type,
                                        self.noise_train,
                                        self.noise_test)
        clean_dir = self.output_dir.replace(noise_task, '')
        best_model = os.path.join(clean_dir, str(self.seed), 'best_model.bin')
        return best_model

    def get_mode(self):
        mode = []
        if DO_TRAIN:
            mode += ['train', 'eval']
        if DO_TEST:
            mode += ['test']
        return ' '.join(mode)

    def get_script(self):
        print('get_script')
        if self.task in ['chid', 'c3', 'cmrc']:
            filename = 'run_mrc_' + self.task + '.sh'
            # print(f'{filename}')
            return os.path.join('scripts', filename)
        elif self.task == 'drcd':
            return os.path.join('scripts', 'run_mrc_cmrc.sh')
        elif self.task == 'cluener':
            return os.path.join('scripts', 'run_ner.sh')
        else:
            return os.path.join('scripts', 'run_finetune.sh')
        
    def set_task_specific_settings(self):
        raise NotImplementedError

    def print_vars(self):
        print(f'self.script = {self.script}')
        print(f'self.task = {self.task}')
        print(f'self.tokenizer = {self.tokenizer}')
        print(f'self.init_checkpoint = {self.init_checkpoint}')
        print(f'self.seed = {self.seed}')
        print(f'self.vocab_file = {self.vocab_file}')
        print(f'self.train_dir = {self.train_dir}')
        print(f'self.dev_dir = {self.dev_dir}')
        print(f'self.test_dir = {self.test_dir}')
        print(f'self.output_dir = {self.output_dir}')
        if self.noise_type is not None:
            print(f'self.test_model = {self.test_model}')

    def get_vars(self):
        ret = {
            'out_dir': self.output_dir,
            'init_checkpoint': self.init_checkpoint,
            'task_name': self.task,
            'config_file': self.config_file,
            'vocab_file': self.vocab_file,
            'vocab_model_file': self.vocab_model_file,
            'tokenizer_type': self.tokenizer_type,
            'data_dir': self.data_dir,
            'train_dir': self.train_dir,
            'dev_dir': self.dev_dir,
            'test_dir': self.test_dir,
            'seed': self.seed,
            'epochs': self.get_epochs(),
            'fewshot': str(int(self.fewshot)),
            # 'convert_to_simplified': self.drcd_convert_to_simplified,
            # 'batch_size': self.batch_size,
            'mode': self.mode,
            'classification_split_char': str(int(self.classification_split_char)),
            'two_level_embeddings': str(int(self.two_level_embeddings)),
            'debug': str(int(self.debug)),
        }
        if self.noise_type is not None:
            ret['test_model'] = self.test_model
        if self.use_cws:
            ret['cws_vocab_file'] = self.cws_vocab_file
        return ret

    def get_cmd(self, script_last=True):
        cmd = []
        cmd += ['out_dir=' + self.output_dir]
        cmd += ['task_name=' + self.task]
        cmd += ['init_checkpoint=' + self.init_checkpoint]
        cmd += ['config_file=' + self.config_file]
        cmd += ['vocab_file=' + self.vocab_file]
        cmd += ['vocab_model_file=' + self.vocab_model_file]
        cmd += ['cws_vocab_file=' + self.cws_vocab_file]
        cmd += ['tokenizer_type=' + self.tokenizer_type]
        cmd += ['train_dir=' + self.train_dir]
        cmd += ['dev_dir=' + self.dev_dir]
        cmd += ['test_dir=' + self.test_dir]
        cmd += ['seed=' + str(self.seed)]
        cmd += ['epochs=' + str(6)]   # TODO: update
        cmd += ['fewshot=' + str(int(self.fewshot))]
        # cmd += ['convert_to_simplified=' + self.drcd_convert_to_simplified]
        # cmd += ['batch_size=' + str()]
        cmd += ['mode=' + self.mode]
        cmd += ['classification_split_char=' + str(int(self.classification_split_char))]
        cmd += ['two_level_embeddings=' + str(int(self.two_level_embeddings))]
        cmd += ['debug=' + str(int(self.debug))]
        cmd += [self.script]
        return cmd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--two_level_embeddings')
    return p.parse_args()


# Change these settings
USE_SLURM = False		# Don't use this anymore
DEBUG = False
DONT_RUN = False
RUN_IN_BG = False
START_FROM_CKPT = False	# Not supported
SLEEP_DURATION = True

TWO_LEVEL_EMBEDDINGS = False
# USE_BASE = False
# USE_LONG = False
USE_SHUFFLED = False
USE_BYTE = True
USE_RANDOM_INDEX = False
USE_NO_INDEX = False
USE_CWS = False
NOISE_TYPE = None
# NOISE_TYPE = 'glyph'
# NOISE_TYPE = 'phonetic'

NOISE_TRAIN = [
    0,
]

NOISE_TEST = [
    # 0,
    # 10,
    # 20,
    # 30,
    # 40,
    50,
    100,
]

SEEDS = [
    10,
    # 11, 12, 
    # 13, 14,
    # 15, 16, 17, 18, 19,
]
TOKENIZERS = [
    # 'cangjie',
    # 'pinyin',
    # 'stroke',
    'wubi',
    # 'zhengma',
    # 'zhuyin',
    # 'raw',
    # 'bert',
    # 'pinyin_concat_wubi',
]

DO_TRAIN = True
DO_TEST = True
FEWSHOT = False
TASKS = [
    'tnews',
    # 'iflytek',
    # 'wsc',
    # 'afqmc',
    # 'csl',
    # 'ocnli',
    # 'cmrc',
    # 'drcd',
    # 'chid',
    # 'c3',
    # 'lcqmc',
    # 'bq',
    # 'thucnews',
    # 'chid',
    # 'cluener',
    # 'chinese_nli' ,  # Hu Hai's ChineseNLIProbing
]


# Assert bound settings
if any(task in ['cmrc', 'drcd', 'cluener'] for task in TASKS):
    assert TWO_LEVEL_EMBEDDINGS
if any(task not in ['cmrc', 'drcd', 'cluener'] for task in TASKS):
    assert not TWO_LEVEL_EMBEDDINGS
if NOISE_TYPE == 'glyph':
    assert NOISE_TEST == [50, 100]
if NOISE_TYPE == 'phonetic':
    assert NOISE_TEST == [10, 20, 30, 40, 50]
assert sum([USE_CWS, USE_BYTE, USE_RANDOM_INDEX]) == 1



def submit_job(task: str, tokenizer: str, ckpt: str, seed: int, **kwargs):
    job = Job(task, tokenizer, ckpt, seed, **kwargs)
    job.print_vars()

    if DONT_RUN:    # Just for testing
        return

    os.makedirs(os.path.join(job.output_dir, str(seed)), exist_ok=True)
    
    script = job.get_script()
    env = job.get_vars()
    # print(env)
    # exit()
    if RUN_IN_BG:
        raise NotImplementedError
    else:
        # Make sure all variables in environment is str, 
        # and bool is "0" or "1".
        for k in env:
            if isinstance(env[k], bool):
                env[k] = str(int(env[k]))
            env[k] = str(env[k])
        env.update(os.environ)
        print('Executing', script)
        process = subprocess.run([script], env=env)

def get_best_ckpt(tokenizer):
    if tokenizer == 'pinyin_concat_wubi':
        return 'ckpt_8840'
    elif USE_NO_INDEX:
        return consts.BEST_CKPTS_NO_INDEX[tokenizer]
    elif USE_SHUFFLED:
        return consts.BEST_CKPTS_SHUFFLED[tokenizer]
    elif USE_CWS:
        return consts.BEST_CKPTS_CWS[tokenizer]
    else:
        return consts.BEST_CKPTS[tokenizer]

def finetune_noisy():
    print("finetune_noisy()")
    for task in TASKS:
        for tokenizer in TOKENIZERS:
            for noise_train in NOISE_TRAIN:
                for noise_test in NOISE_TEST:
                    ckpt = get_best_ckpt(tokenizer)
                    for seed in SEEDS:
                        submit_job(
                            task, tokenizer, ckpt, seed,
                            debug=DEBUG, 
                            two_level_embeddings=TWO_LEVEL_EMBEDDINGS,
                            use_base=USE_BASE,
                            use_long=USE_LONG,
                            use_shuffled=USE_SHUFFLED,
                            use_no_index=USE_NO_INDEX,
                            noise_type=NOISE_TYPE,
                            noise_train=noise_train,
                            noise_test=noise_test,
                            use_sp=USE_SP,
                            classification_split_char=CLASSIFICATION_SPLIT_CHAR)

def finetune():
    '''
    Ordinary finetuning, not noisy task.
    '''
    for task in TASKS:
        for tokenizer in TOKENIZERS:
            ckpt = get_best_ckpt(tokenizer)
            for seed in SEEDS:
                submit_job(task, tokenizer, ckpt, seed,
                           debug=DEBUG, 
                           two_level_embeddings=TWO_LEVEL_EMBEDDINGS,
                        #    use_base=USE_BASE,
                        #    use_long=USE_LONG,
                           use_shuffled=USE_SHUFFLED,
                           use_no_index=USE_NO_INDEX,
                        #    use_sp=USE_SP,
                           use_cws=USE_CWS,
                        #    classification_split_char=CLASSIFICATION_SPLIT_CHAR
                           )


def main():
    if NOISE_TYPE != None:
        finetune_noisy()
    else:
        finetune()


if __name__ == '__main__':
    main()
