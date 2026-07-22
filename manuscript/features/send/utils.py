import os
import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

def evaluate_rmse(eval_pred):
    y_pred, y_true = eval_pred
    rmse = root_mean_squared_error(y_true, y_pred)
    return {"rmse": rmse}


def rmse_loss(y_true, y_pred):
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    return torch.sqrt(torch.mean((y_true-y_pred)**2))


def compute_metrics(eval_pred):
    y_pred, y_true = eval_pred
    return {"RMSE": rmse_loss(y_true, y_pred)}


def ccc(y_true, y_pred):
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)
    X = torch.stack((y_pred, y_true))
    true_mean = torch.mean(y_true)
    true_var = torch.var(y_true, correction=0)
    pred_mean =  torch.mean(y_pred)
    pred_var = torch.var(y_pred, correction=0)
    covar = torch.cov(X, correction=0)[0][1]
    ccc = float(2*covar / (true_var + pred_var +  (pred_mean-true_mean)**2))
    return ccc


def ccc_pvalue(y_true, y_pred, n_permutations=1000, seed=13):
    torch.manual_seed(seed)
    observed_ccc = ccc(y_true, y_pred)
    y_pred_tensor = torch.Tensor(y_pred)
    len_y_pred = len(y_pred)
    permuted_cccs = []
    for i in range(n_permutations):
        permuted_indices = torch.randperm(len_y_pred)
        permuted_y_pred = y_pred_tensor[permuted_indices]
        permuted_cccs += [ccc(y_true, permuted_y_pred)]
    p_value = float(torch.sum(torch.Tensor(permuted_cccs) >= observed_ccc) / n_permutations)
    return p_value


def run_evaluation(model_path, trainer, split, inputs, **kwargs):

    predictions = trainer.predict(inputs[split]).predictions[:,0]
    rmse_result = root_mean_squared_error(inputs[split]["label"], predictions)
    r_result, r_p = pearsonr(inputs[split]["label"], predictions)
    ccc_result = ccc(inputs[split]["label"], predictions)
    ccc_p = ccc_pvalue(inputs[split]["label"], predictions)

    print(f"{split.upper():5s}   Mean: {np.mean(predictions):6.5f}   SD: {np.std(predictions):6.5f}   RMSE: {rmse_result:6.5f}   R: {r_result:6.5f} (p={r_p:6.5f})   CCC: {ccc_result:6.5f} (p={ccc_p:6.5f})")
    
    return [predictions, 
            {"path": model_path, 
            "split": split,
            **kwargs,
            "pred_mean": np.mean(predictions),
            "pred_sd": np.std(predictions),
            "rmse": rmse_result,
            "r": r_result,
            "r_p": r_p,
            "ccc": ccc_result,
            "ccc_p": ccc_p}]


def run_finetuning(model_path, trainer, param_dict, inputs, plot=True, splits=["train", "valid", "test"]):
    
    if not os.path.exists(f"{model_path}/model.safetensors"):
        print(f"\nTRAINING MODEL\nPath: {model_path}\n")
        trainer.train()
        trainer.save_model(model_path)

    print(f"EVALUATING MODEL")
    evaluation = []
    for split in splits:
        split_predictions, split_evaluation = run_evaluation(model_path, trainer, split, inputs, **param_dict)
        evaluation += [split_evaluation]
        if plot:
            plot_predictions(inputs[split]["label"], split_predictions, f"{model_path.split('/')[-1]}_{split}")
    print("\n" + "-"*100 + "\n")
    
    return model_path, evaluation


def plot_predictions(x, y, plot_name):

    font_path = "style/cmu/cmunss.ttf"
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    
    r, r_p = pearsonr(x, y)
    ccc_result = ccc(x, y)
    ccc_p = ccc_pvalue(x, y, n_permutations=1000)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    
    sns.regplot(x=x, y=y, x_ci="ci", ci=None, 
                line_kws=dict(color="black", linewidth=1), 
                scatter_kws=dict(s=25, color="gray", edgecolor="none", alpha=0.4))

    plt.text(0.05, 0.95, f"r={r:4.2f} (p={r_p:4.2f})", font=font_manager.FontProperties(fname=font_path, size=14))
    plt.text(0.05, 0.865, f"CCC={ccc_result:4.2f} (p={ccc_p:4.2f})", font=font_manager.FontProperties(fname=font_path, size=14))
    
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    
    ax.set_xlabel("Labels", labelpad=10, font=font_manager.FontProperties(fname=font_path, size=20))
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks, font=font_manager.FontProperties(fname=font_path, size=14))
    
    ax.set_ylabel("Predictions", labelpad=10, font=font_manager.FontProperties(fname=font_path, size=20))
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, font=font_manager.FontProperties(fname=font_path, size=14))
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.savefig(f"figures/{plot_name}.png", dpi=300, bbox_inches="tight")
    plt.show()
