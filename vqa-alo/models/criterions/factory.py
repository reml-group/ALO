from bootstrap.lib.options import Options
from block.models.criterions.vqa_cross_entropy import VQACrossEntropyLoss
from .rubi_criterion_batch import RUBiCriterionBatch
from .cfvqa_criterion import CFVQACriterion
from .cfvqaintrod_criterion import CFVQAIntroDCriterion
from .rubiintrod_criterion import RUBiIntroDCriterion
from .rubi_criterion import RUBiCriterion
from .cfvqa_criterion_batch import CFVQACriterionBatch
from .ordinary_criterion import OrdinaryCriterion
from .ordinary_criterion_batch import OrdinaryCriterionBatch

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split
    eval_only = 'train' not in engine.dataset
    
    opt = Options()['model.criterion']
    if split == "test" and 'tdiuc' not in Options()['dataset.name']:
        return None
    if name == 'vqa_cross_entropy':
        criterion = VQACrossEntropyLoss()
    elif name == "rubi_criterion_batch":
        criterion = RUBiCriterionBatch(
            question_loss_weight=opt['question_loss_weight'],
            loose_batch_num=opt['loose_batch_num']
        )
    elif name == 'ordinary_criterion':
        criterion = OrdinaryCriterion()
    elif name == "cfvqa_criterion":
        criterion = CFVQACriterion(
            question_loss_weight=opt['question_loss_weight'],
            vision_loss_weight=opt['vision_loss_weight'],
            is_va=True,
        )
    elif name == "cfvqasimple_criterion":
        criterion = CFVQACriterion(
            question_loss_weight=opt['question_loss_weight'],
            is_va=False,
        )
    elif name == "cfvqasimple_criterion_batch":
        criterion = CFVQACriterionBatch(
            question_loss_weight=opt['question_loss_weight'],
            is_va=False,
            loose_batch_num=opt['loose_batch_num']
        )
    elif name == "cfvqaintrod_criterion":
        criterion = CFVQAIntroDCriterion()
    elif name == "rubiintrod_criterion":
        criterion = RUBiIntroDCriterion()
    elif name == "cfvqa_criterion":
        criterion = CFVQACriterion(
            question_loss_weight=opt['question_loss_weight'],
            vision_loss_weight=opt['vision_loss_weight'],
            is_va=True,
            gamma=opt['gamma']
        )
    elif name == "cfvqasimple_criterion":
        criterion = CFVQACriterion(
            question_loss_weight=opt['question_loss_weight'],
            is_va=False,
            gamma=opt['gamma']
        )
    elif name == "rubi_criterion":
        criterion = RUBiCriterion(question_loss_weight=opt['question_loss_weight'])
    #cfvqa_criterion_batch
    elif name == "cfvqa_criterion_batch":
        criterion = CFVQACriterionBatch(
            question_loss_weight=opt['question_loss_weight'],
            vision_loss_weight=opt['vision_loss_weight'],
            is_va=True,
            loose_batch_num=opt['loose_batch_num']
        )
    elif name == 'ordinary_criterion_batch':
        criterion = OrdinaryCriterionBatch(
            loose_batch_num=opt['loose_batch_num']
        )
    else:
        raise ValueError(name)
    return criterion
