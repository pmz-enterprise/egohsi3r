"""Recurrent inference only; no losses or training loop."""
import torch

@torch.no_grad()
def inference_recurrent_lighter(groups, model, device, verbose=True, is_naive=False, use_ttt3r=False):
    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    with torch.cuda.amp.autocast(enabled=False):
        preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True, use_ttt3r=use_ttt3r
            )
        res = dict(views=batch, pred=preds)
    return res, state_args

