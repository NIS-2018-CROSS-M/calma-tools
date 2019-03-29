import re
import sys
from collections import defaultdict as dd
from random import shuffle


class MLUtil:
    lower_bound=0.8
    num=3
    
    def __init__(self, prediction_params, dataModifyer, nbestModifyer):
        self.dataModifyer = dataModifyer
        self.nbestModifyer = nbestModifyer
        self.prediction_params = prediction_params


    def generate_data(self, orig_data_fn, res_src_fn, res_tgt_fn):
        return self._generate_onmt_data(orig_data_fn, res_src_fn, res_tgt_fn, self.dataModifyer)

    def train(self, train_res_src_fn, train_res_tgt_fn, val_res_src_fn, val_res_tgt_fn, save_model_fn, train_params):
        data_fn = save_model_fn + "-prepared_training_data" #f"onmt-data/{lang}-track{track}"
        self._initialize_data(train_res_src_fn, train_res_tgt_fn, val_res_src_fn, val_res_tgt_fn, data_fn)

        train_params.extend([f"-data {data_fn}", f"-save_model {save_model_fn}"])
        self._train_ml(train_params)
        get_ipython().system('mv {save_model_fn}_step_{train_steps}.pt {save_model_fn}')


    def predict(self, model_filename, input_data_filename, covered_filename, chosen_output_filename):
        output_data_filename = f"{input_data_filename}.out"
        nbest_output_filename = f"{input_data_filename}.nbest.out"
        self.prediction_params.extend([
            f"-model {model_filename}",
            f"-src {input_data_filename}",
            f"-output {output_data_filename}"
        ])
        self._generate_predictions(self.prediction_params, nbest_output_filename)
        nbest_output_modified_filename = nbest_output_filename+"-modified"
        self.modify_nbest(nbest_output_filename, nbest_output_modified_filename, self.nbestModifyer)
        self._choose_best_predictions(nbest_output_modified_filename, covered_filename, chosen_output_filename)

    @staticmethod
    def _generate_onmt_data(fn, res_src_fn, res_tgt_fn, dataModifyer):
        # the method called for each non-processed training data row
        def get_data_entry(language, wordform, lemma, pos_tag, morphological_analysis):
            lemma = ' '.join(lemma)
            wordform = ' '.join(wordform)
            morphological_analysis = morphological_analysis.split('|')
            return wordform, '%s %s' % (
                lemma, ' '.join(['+%s' % x for x in [pos_tag] + morphological_analysis + ["Language=%s" % language]]))

        modify_src_line = dataModifyer.modify_src_line
        modify_tgt_line = dataModifyer.modify_tgt_line
        restore_orig_src_line = dataModifyer.restore_src_line
        restore_orig_tgt_line = dataModifyer.restore_tgt_line

        analyses = dd(set)

        for line in open(fn, encoding='utf-8'):
            line = line.rstrip('\n').rstrip('\r')
            lang, wf, lemma, pos, msd = line.split('\t')
            wf, a = get_data_entry(lang, wf, lemma, pos, msd)
            analyses[wf].add(a)

        tmp_src_fn = res_src_fn + "-default"
        tmp_tgt_fn = res_tgt_fn + "-default"

        tmp_src = open(tmp_src_fn, 'w')
        tmp_tgt = open(tmp_tgt_fn, 'w')
        res_src = open(res_src_fn, 'w')
        res_tgt = open(res_tgt_fn, 'w')

        analyses = list(analyses.items())
        shuffle(analyses)

        for wf, analysis in analyses:
            for a in analysis:
                print(wf, file=tmp_src)
                print(a, file=tmp_tgt)
                print(modify_src_line(wf), file=res_src)
                print(modify_tgt_line(a), file=res_tgt)

    @staticmethod
    def modify_nbest(nbest_src_filename, nbest_tgt_filename, nbestModifyer):
        with open(nbest_src_filename, 'r', encoding='utf-8') as src_f, open(nbest_tgt_filename, 'w',
                                                                            encoding='utf-8') as tgt_f:

            for line in src_f.readlines():
                line = line.rstrip('\n').rstrip('\r')
                if line.startswith("SENT "):
                    line = nbestModifyer.sent_to_baseline_compatible(line)
                elif re.match("^\[[\-\+]?\d+\.\d+\]\s\[", line):
                    line = nbestModifyer.hyp_to_baseline_compatible(line)

                print(line, file=tgt_f)

    @staticmethod
    def _initialize_data(train_src, train_tgt, valid_src, valid_tgt, prepared_training_data_prefix):
        prepr_params = f"-train_src {train_src} -train_tgt {train_tgt} -valid_src {valid_src} -valid_tgt {valid_tgt} -save_data {prepared_training_data_prefix}"
        get_ipython().system(f'/usr/bin/python3 ~/OpenNMT-py/preprocess.py {prepr_params}')

    @staticmethod
    def _train_ml(train_params):
        train_params = " ".join(train_params)
        get_ipython().system(f'/usr/bin/python3 ~/OpenNMT-py/train.py {train_params}')

    @staticmethod
    def _generate_predictions(generation_params, output_filename):
        generation_params = " ".join(generation_params)
        get_ipython().system(f'/usr/bin/python3 ~/OpenNMT-py/translate.py {generation_params} > {output_filename}')

    def _choose_best_predictions(self, nbest_filename, covered_filename, output_filename):
        get_ipython().system(' | '.join([
            f'cat {nbest_filename}',
            'grep -v -P "^\\s+"',
            'grep -v -P "^\\+"',
            '/usr/bin/python3 scripts/get-analyses.py' + ' ' + f'{self.lower_bound} {self.num}' + ' ' + f'{covered_filename} > {output_filename}'
        ]))

    @staticmethod
    def score_predictions(res_file_fn, gold_file_fn, output_fn, dataEvaluator):

        def readdata(fn):
            data = {otype: dd(set) for otype in dataEvaluator.otypes}
            for line in open(fn):
                line = line.strip('\n')
                if line:
                    data = dataEvaluator.update_data(data, line)
            return data

        sysdata = readdata(res_file_fn)
        golddata = readdata(gold_file_fn)

        output_f = open(output_fn, 'a+', encoding='utf-8')
        for otype in dataEvaluator.otypes:
            tp = 0
            fp = 0
            fn = 0
            for wf in sysdata[otype]:
                tp += len(sysdata[otype][wf] & golddata[otype][wf])
                fp += len(sysdata[otype][wf] - golddata[otype][wf])
                fn += len(golddata[otype][wf] - sysdata[otype][wf])
            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            fscore = 2 * recall * precision / (recall + precision)
            print("Recall for %s: %.2f" % (otype, recall * 100), file=output_f)
            print("Precision for %s: %.2f" % (otype, precision * 100), file=output_f)
            print("F1-score for %s: %.2f" % (otype, fscore * 100), file=output_f)
            print("", file=output_f)

        output_f.close()
