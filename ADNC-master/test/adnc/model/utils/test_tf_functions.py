# Copyright 2018 Jörg Franke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import pytest
import tensorflow as tf

from adnc.model.utils import oneplus


@pytest.fixture()
def session():
    with tf.Session() as sess:
        yield sess
    tf.reset_default_graph()


def test_oneplus(session):
    tf_x = tf.constant([1, 2, 3], dtype=tf.float32, )
    tf_x_oneplus = oneplus(tf_x)
    np_x_oneplus = tf_x_oneplus.eval()

    assert np.allclose(np_x_oneplus, 1 + np.log1p(np.exp([1, 2, 3])))
