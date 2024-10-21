# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Directional Bias Amplification metric."""

import evaluate
import datasets

import numpy as np

_DESCRIPTION = """
Directional Bias Amplification is a metric that captures the amount of bias (i.e., a conditional probability) that is amplified. 
This metric was introduced in the ICML 2021 paper "Directional Bias Amplification" (https://arxiv.org/abs/2102.12594).
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions (`array` of `int`): Predicted task labels. Array of size n x |T|. n is number of samples, |T| is number of task labels. All values are binary 0 or 1.
    references (`array` of `int`): Ground truth task labels. Array of size n x |T|. n is number of samples, |T| is number of task labels.  All values are binary 0 or 1.
    attributes(`array` of `int`): Ground truth attribute labels. Array of size n x |A|. n is number of samples, |A| is number of attribute labels.  All values are binary 0 or 1.
Returns
    bias_amplification(`float`): Bias amplification value. Minimum possible value is 0, and maximum possible value is 1.0. The higher the value, the more "bias" is amplified.
    disagg_bias_amplification (`array` of `float`): Array of size (number of unique attribute label values) x (number of unique task label values). Each array value represents the bias amplification of that particular task given that particular attribute.
"""


_CITATION = """
@inproceedings{wang2021biasamp,
author = {Angelina Wang and Olga Russakovsky},
title = {Directional Bias Amplification},
booktitle = {International Conference on Machine Learning (ICML)},
year = {2021}
}
"""


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class DirectionalBiasAmplification(evaluate.EvaluationModule):
    def _info(self):
        return evaluate.EvaluationModuleInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                    "attributes": datasets.Sequence(datasets.Value("int32")),
                }
            ),
            reference_urls=["https://arxiv.org/abs/2102.12594"],
        )

    def _compute(self, predictions, references, attributes):

        task_preds, task_labels, attribute_labels = (
            np.array(predictions),
            np.array(references),
            np.array(attributes),
        )

        assert (
            len(task_labels.shape) == 2 and len(attribute_labels.shape) == 2
        ), 'Please read the shape of the expected inputs, which should be "num samples" by "num classification items"'
        assert (
            len(task_labels) == len(attribute_labels) == len(task_preds)
        ), "Please make sure the number of samples in the three input arrays is the same."

        num_t, num_a = task_labels.shape[1], attribute_labels.shape[1]

        # only include images that have attribute(s) and task(s) associated with it
        keep_indices = np.array(
            list(
                set(np.where(np.sum(task_labels, axis=1) > 0)[0]).union(
                    set(np.where(np.sum(attribute_labels, axis=1) > 0)[0])
                )
            )
        )
        task_labels_ind, attribute_labels_ind = (
            task_labels[keep_indices],
            attribute_labels[keep_indices],
        )

        # y_at calculation
        p_at = np.zeros((num_a, num_t))
        p_a_p_t = np.zeros((num_a, num_t))
        num = len(task_labels)
        for a in range(num_a):
            for t in range(num_t):
                t_indices = np.where(task_labels_ind[:, t] == 1)[0]
                a_indices = np.where(attribute_labels_ind[:, a] == 1)[0]
                at_indices = set(t_indices) & set(a_indices)
                p_a_p_t[a][t] = (len(t_indices) / num) * (len(a_indices) / num)
                p_at[a][t] = len(at_indices) / num
        y_at = np.sign(p_at - p_a_p_t)

        # delta_at calculation
        t_cond_a = np.zeros((num_a, num_t))
        that_cond_a = np.zeros((num_a, num_t))
        for a in range(num_a):
            for t in range(num_t):
                t_cond_a[a][t] = np.mean(
                    task_labels[:, t][np.where(attribute_labels[:, a] == 1)[0]]
                )
                that_cond_a[a][t] = np.mean(
                    task_preds[:, t][np.where(attribute_labels[:, a] == 1)[0]]
                )
        delta_at = that_cond_a - t_cond_a

        values = y_at * delta_at
        val = np.nanmean(values)

        val, values
        return {"bias_amplification": val, "disagg_bias_amplification": values}


if __name__ == "__main__":

    # Data Initialization
    from utils.datacreator import dataCreator

    P, D, D2, M1, M2 = dataCreator(16384, 0.2, False, 0.05)
    P = P.reshape(-1, 1)
    D = D.reshape(-1, 1)
    D2 = D2.reshape(-1, 1)
    M1 = M1.reshape(-1, 1)
    M2 = M2.reshape(-1, 1)

    # Calculating Params
    model_1_acc = np.sum(D == M1) / D.shape[0]
    model_2_acc = np.sum(D == M2) / D.shape[0]

    # Parameter Initialization
    dba = DirectionalBiasAmplification()

    dba_1 = dba._compute(M1, D, P)
    print(f"DBA for case 1: {dba_1}")
    print("______________________________________")
    print("______________________________________")
    dba_2 = dba._compute(M2, D, P)
    print(f"DBA for case 2: {dba_2}")
    print("______________________________________")
    print("______________________________________")
    dba_3 = dba._compute(M1, D2, P)
    print(f"DBA for case 3: {dba_3}")
    print("______________________________________")
    print("______________________________________")
    dba_4 = dba._compute(M2, D2, P)
    print(f"DBA for case 4: {dba_4}")
