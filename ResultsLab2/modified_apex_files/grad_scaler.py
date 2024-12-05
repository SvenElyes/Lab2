# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
from collections import defaultdict

import torch

from apex.transformer import parallel_state
from apex.amp import scale_loss

class GradScaler:
    """
    Gradient scaler for model-parallel inf check using Apex AMP.
    The inf in gradients are checked across tensor-parallel ranks in 
    (1) executing optimizer step and (2) gradient scaler update.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def scale(self, loss, optimizer):
        """
        Scales the loss using Apex AMP's scale_loss function.
        """
        if self.enabled:
            return scale_loss(loss, optimizer)
        return loss

    def unscale_(self, optimizer):
        """
        Placeholder for unscaling gradients. Not required with Apex AMP.
        """
        pass

    def step(self, optimizer):
        """
        Performs the optimizer step if no NaN or Inf gradients are found.
        """
        retval = None

        # Check for NaNs or Infs in gradients
        found_inf = any(
            torch.isinf(param.grad).any() or torch.isnan(param.grad).any()
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        )

        # Synchronize across model parallel instances
        found_inf_tensor = torch.tensor([float(found_inf)], device=torch.device("cuda"))
        torch.distributed.all_reduce(
            found_inf_tensor, op=torch.distributed.ReduceOp.MAX, group=parallel_state.get_model_parallel_group()
        )

        if found_inf_tensor.item() == 0:
            retval = optimizer.step()  # Perform optimizer step only if gradients are valid

        return retval

    def update(self):
        """
        Placeholder for scale factor update. Apex AMP handles scaling internally.
        """
        pass
