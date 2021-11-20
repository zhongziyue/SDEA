import os

import torch as t

from config.KBConfig import *
from tools.Announce import Announce


class ModelTools:
    def __init__(self, patient, min_max='min'):
        self.min_max = min_max
        if min_max == 'min':
            self.bst_score = float('inf')
        else:
            assert min_max == 'max', 'unknown min_max'
            self.bst_score = float('-inf')
        self.patient = patient
        self.current_patient = 0

    def early_stopping(self, model: t.nn.Module, path, score):
        save = False
        if self.min_max == 'min':
            if score < self.bst_score:
                save = True
        else:
            if score > self.bst_score:
                save = True
        if save:
            print(Announce.printMessage(), 'score:', self.bst_score, '->', score)
            self.bst_score = score
            ModelTools.save_model(model, path)
            # ModelTools.save_state(model)

            self.current_patient = 0
            return False
        else:
            self.current_patient += 1
            print(Announce.printMessage(), 'bst score:', self.bst_score, 'current score:', score, 'patient:', self.current_patient)
            if self.current_patient >= self.patient:
                return True
            else:
                return False

    @staticmethod
    def save_model(model: t.nn.Module, output_sub_dir) -> None:
        # output_sub_dir = os.path.join(KBConfig.result_path, KBConfig.result_name)
        # output_sub_dir = links.model_save
        print(Announce.printMessage(), '保存模型:', output_sub_dir)
        model_to_save = model.module if hasattr(model, 'module') else model
        fold = os.path.dirname(output_sub_dir)
        if not os.path.exists(fold):
            os.makedirs(fold)
        t.save(model_to_save, output_sub_dir)

        # model_to_save.save_pretrained(output_sub_dir)

    # @staticmethod
    # def save_state(model: t.nn.Module):
    #     model_path = FileNameTools.mode_state_path()
    #     print(Announce.printMessage(), '保存模型参数:', model_path)
    #     t.save(model.state_dict(), model_path)
    #
    @staticmethod
    def load_model(model_path):
        # model_path = FileNameTools.model_path()
        return t.load(model_path)
    #
    # @staticmethod
    # def load_state(model: t.nn.Module):
    #     model_path = FileNameTools.mode_state_path()
    #     print(Announce.printMessage(), 'load state:', model_path)
    #     model.load_state_dict(t.load(model_path))
