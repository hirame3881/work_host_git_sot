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
from adnc.model.utils import EarlyStop


def test_early_stop():
    test_seq = [1, 2, 1, 4, 2, 1, 3, 4, 5, 2, 7, 2, 3, 4, 5]
    true_seq = [False, False, False, False, False, False, False, False, True, False, False, False, False, False, True]

    e_stop = EarlyStop(4)
    for test, true in zip(test_seq, true_seq):
        assert true == e_stop(test)
