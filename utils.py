from style import style

import os
import scipy
import numpy as np
import pandas as pd

from statsmodels.stats.multitest import multipletests


########################################################################
############################ DATA MANAGEMENT ###########################
########################################################################

def list_files(path):
    
    files = sorted([file for file in os.listdir(path) if not file.startswith(".")])
    
    return files


def load_vas_df(filename):
    
    vas_excluded_vars = ["vas_time", "vas_freeresponse", "vasrating_number", "vas_ratings_and_free_response_complete"]
    vas_df = pd.read_csv(filename, index_col=0)
    vas_vars = [var for var in vas_df.columns if var.startswith("vas") and not var in vas_excluded_vars]
    
    return vas_df


########################################################################
############################## STATISTICS ##############################
########################################################################

############################## FORMATTING ##############################

def sci_notation(number, sig_fig=2):
    
    ret_string = "{0:.{1:d}e}".format(number, sig_fig)
    a, b = ret_string.split("e")
    b = int(b)
    sci_not = a + "x10" + str(b)
    
    return sci_not


def p2sig(p):
    sig = ""
    if p < 0.1:
        sig = "+"
    if p < 0.05:
        sig = "*"
    if p < 0.01:
        sig = "**"
    if p < 0.001:
        sig = "***"
    return sig


############################# CALCULATING ##############################

def compute_U_CLES(U, group1, group2):
    
    n1 = len(group1)
    n2 = len(group2)
    effect_size = U / (n1 * n2)
    CLES = max(effect_size, 1 - effect_size)
    
    return CLES


def compute_demo_differences(demo_dfs, demo_stat_path, 
                             vars=["Age", "Gender", "Education", "ehi_handedness", 
                                   "Race", "pclc_total", "PHQ9", "GAD7", "ELS"], 
                             numerical=["Age", "pclc_total", "PHQ9", "GAD7", "ELS"], 
                             categorical=["Gender", "Education", "ehi_handedness", "Race"],
                             conditions=["placebo", "mdma_high"], verbose=True):
    
    for condition, df in demo_dfs.items():
        print(condition.upper())
        stat_df = pd.DataFrame()
        for var in vars:
            if var in numerical:
                vals_low = df[df["Group"] == "low"][var]
                vals_high = df[df["Group"] == "high"][var]
                stat, p = scipy.stats.mannwhitneyu(vals_low, vals_high)
            if var in categorical:
                vals_low = pd.Series([np.nan])
                vals_high = pd.Series([np.nan])
                contingency = pd.crosstab(df["Group"], df[var])
                stat, p, dof, expected_freq = scipy.stats.chi2_contingency(contingency)
            row_df = pd.DataFrame({"condition": condition, "var": var, 
                                   "n_low": len(vals_low), "med_low": vals_low.median(), 
                                   "q1_low": vals_low.quantile(0.25), "q3_low": vals_low.quantile(0.75),
                                   "n_high": len(vals_high), "med_high": vals_high.median(), 
                                   "q1_high": vals_high.quantile(0.25), "q3_high": vals_high.quantile(0.75),
                                   "stat": stat, "p": p, "sig": p2sig(p)}, index=[0])
            stat_df = pd.concat([stat_df, row_df])
        stat_df.to_csv(f"{demo_stat_path}_{condition}.csv", index=None)
        if verbose:
            print(stat_df)
            print()
            
    return stat_df


def compute_group_differences(df, vars,
                              conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                               verbose=True):
    
    stat_dfs = {}
    for condition in conditions:
        stat_rows = []
        for var in vars:
            vals_low = df[(df["biotype"] == "low") & (df["condition"] == condition)][var].dropna()
            vals_high = df[(df["biotype"] == "high") & (df["condition"] == condition)][var].dropna()
            U, p = scipy.stats.mannwhitneyu(vals_low, vals_high)
            CLES = compute_U_CLES(U, vals_low, vals_high)
            var_row = {"var": var,
                       "n_low": len(vals_low),
                       "μ_low": vals_low.mean(),
                       "med_low": vals_low.median(),
                       "min_low": vals_low.min(),
                       "max_low": vals_low.max(),
                       "q1_low": vals_low.quantile(0.25),
                       "q3_low": vals_low.quantile(0.75),
                       "n_high": len(vals_high),
                       "μ_high": vals_high.mean(),
                       "med_high": vals_high.median(),
                       "min_high": vals_high.min(),
                       "max_high": vals_high.max(),
                       "q1_high": vals_high.quantile(0.25),
                       "q3_high": vals_high.quantile(0.75),
                       "U": U,
                       "CLES": CLES,
                       "p": p}
            stat_rows += [var_row]
        stat_df = pd.DataFrame(stat_rows, index=vars)
        _, fdrs, _, _ = multipletests(stat_df["p"], method="fdr_bh")
        stat_df["FDR"] = fdrs
        stat_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        stat_dfs[condition] = stat_df
        if verbose:
            print("-"*100 + f"\nBIOTYPE COMPARISONS BY {condition.upper()} CONDITION\n" + "-"*100)
            print(stat_df)
            print()
        
    for biotype in biotypes:
        stat_rows = []
        for var in vars:
            vals_placebo = df[(df["biotype"] == biotype) & (df["condition"] == "placebo")][var].dropna()
            vals_mdma = df[(df["biotype"] == biotype) & (df["condition"] == "mdma_high")][var].dropna()
            U, p = scipy.stats.mannwhitneyu(vals_placebo, vals_mdma) 
            CLES = compute_U_CLES(U, vals_placebo, vals_mdma)
            var_row = {"var": var,
                       "n_placebo": len(vals_placebo),
                       "μ_placebo": vals_placebo.mean(),
                       "med_placebo": vals_placebo.median(),
                       "min_placebo": vals_placebo.min(),
                       "max_placebo": vals_placebo.max(),
                       "q1_placebo": vals_placebo.quantile(0.25),
                       "q3_placebo": vals_placebo.quantile(0.75),
                       "n_mdma": len(vals_mdma),
                       "μ_mdma": vals_mdma.mean(),
                       "med_mdma": vals_mdma.median(),
                       "min_mdma": vals_mdma.min(),
                       "max_mdma": vals_mdma.max(),
                       "q1_mdma": vals_mdma.quantile(0.25),
                       "q3_mdma": vals_mdma.quantile(0.75),
                       "U": U,
                       "CLES": CLES,
                       "p": p}
            stat_rows += [var_row]
        stat_df = pd.DataFrame(stat_rows, index=vars)
        _, fdrs, _, _ = multipletests(stat_df["p"], method="fdr_bh")
        stat_df["FDR"] = fdrs
        stat_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        stat_dfs[biotype] = stat_df
        if verbose:
            print("-"*100 + f"\nCONDITION COMPARISONS BY {biotype.upper()} BIOTYPE\n" + "-"*100)
            print(stat_df)
            print()

    return stat_dfs


def compute_group_block_differences(df, vars, n_blocks=5, 
                                    conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                                    verbose=True):
    
    stat_dfs = {}
    for condition in conditions:
        stat_rows = []
        for var in vars:
            var_rows = []
            for block in range(1, n_blocks+1):
                vals_low = df[(df["biotype"] == "low") & (df["condition"] == condition) & (df["block"] == block)][var].dropna()
                vals_high = df[(df["biotype"] == "high") & (df["condition"] == condition) & (df["block"] == block)][var].dropna()
                U, p = scipy.stats.mannwhitneyu(vals_low, vals_high)
                CLES = compute_U_CLES(U, vals_low, vals_high)
                var_row = {"var": var,
                           "block": block,
                           "n_low": len(vals_low),
                           "μ_low": vals_low.mean(),
                           "med_low": vals_low.median(),
                           "min_low": vals_low.min(),
                           "max_low": vals_low.max(),
                           "q1_low": vals_low.quantile(0.25),
                           "q3_low": vals_low.quantile(0.75),
                           "n_high": len(vals_high),
                           "μ_high": vals_high.mean(),
                           "med_high": vals_high.median(),
                           "min_high": vals_high.min(),
                           "max_high": vals_high.max(),
                           "q1_high": vals_high.quantile(0.25),
                           "q3_high": vals_high.quantile(0.75),
                           "U": U,
                           "CLES": CLES,
                           "p": p}
                stat_rows += [var_row]
        stat_df = pd.DataFrame(stat_rows)
        _, fdrs, _, _ = multipletests(stat_df["p"], method="fdr_bh")
        stat_df["FDR"] = fdrs
        stat_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        stat_dfs[condition] = stat_df
        if verbose:
            print("-"*100 + f"\nBIOTYPE COMPARISONS BY {condition.upper()} CONDITION\n" + "-"*100)
            print(stat_df)
            print()
        
    for biotype in biotypes:
        stat_rows = []
        for var in vars:
            var_df = pd.DataFrame(index=range(1, n_blocks+1))
            for block in range(1, n_blocks+1):
                block_df = pd.DataFrame(index=vars)
                vals_placebo = df[(df["biotype"] == biotype) & (df["condition"] == "placebo") & (df["block"] == block)][var].dropna()
                vals_mdma = df[(df["biotype"] == biotype) & (df["condition"] == "mdma_high") & (df["block"] == block)][var].dropna()
                U, p = scipy.stats.mannwhitneyu(vals_placebo, vals_mdma) 
                CLES = compute_U_CLES(U, vals_placebo, vals_mdma)
                var_row = {"var": var,
                           "block": block,
                           "n_placebo": len(vals_placebo),
                           "μ_placebo": vals_placebo.mean(),
                           "med_placebo": vals_placebo.median(),
                           "min_placebo": vals_placebo.min(),
                           "max_placebo": vals_placebo.max(),
                           "q1_placebo": vals_placebo.quantile(0.25),
                           "q3_placebo": vals_placebo.quantile(0.75),
                           "n_mdma": len(vals_mdma),
                           "μ_mdma": vals_mdma.mean(),
                           "med_mdma": vals_mdma.median(),
                           "min_mdma": vals_mdma.min(),
                           "max_mdma": vals_mdma.max(),
                           "q1_mdma": vals_mdma.quantile(0.25),
                           "q3_mdma": vals_mdma.quantile(0.75),
                           "U": U,
                           "CLES": CLES,
                           "p": p}
                stat_rows += [var_row]
        stat_df = pd.DataFrame(stat_rows)
        _, fdrs, _, _ = multipletests(stat_df["p"], method="fdr_bh")
        stat_df["FDR"] = fdrs
        stat_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        stat_dfs[biotype] = stat_df
        if verbose:
            print("-"*100 + f"\nCONDITION COMPARISONS BY {biotype.upper()} BIOTYPE\n" + "-"*100)
            print(stat_df)
            print()

    return stat_dfs


def compute_correlation_CI(df, var_x, var_y, stat=scipy.stats.spearmanr, 
                           CI_width=0.95, bootstrap_n_iter=1000, seed=9):

    np.random.seed(seed)
    index = range(len(df))
    distribution = []
    for i in range(bootstrap_n_iter):
        index_resampled = np.random.choice(index, size=len(index), replace=True)
        boot_df = df.iloc[index_resampled]
        boot_r, boot_p = stat(boot_df[var_y], boot_df[var_x])
        distribution += [boot_r]
    CI_idx = int((1.0 - CI_width) * bootstrap_n_iter) 
    distribution = sorted(distribution)
    CI_lower = distribution[CI_idx]
    CI_upper = distribution[-1 * CI_idx]

    return CI_lower, CI_upper


def compute_correlation(df, var_x, var_y, condition,
                        stat=scipy.stats.spearmanr, 
                        CI_width=0.95, bootstrap_n_iter=1000, seed=9):

    r, p = stat(df[var_y], df[var_x])
    CI_lower, CI_upper = compute_correlation_CI(df, var_x, var_y, 
                                                stat=stat, CI_width=CI_width, 
                                                bootstrap_n_iter=bootstrap_n_iter,
                                                seed=seed)
    cor_row = {"condition": condition, 
               "var_x": var_x, 
               "var_y": var_y, 
               "n": len(df), 
               "r": r, 
               "CI_lower": CI_lower, 
               "CI_upper": CI_upper, 
               "p": p}
    
    return cor_row
        

def compute_session_correlation(speech_line_df, var_x, vars_y,
                                stratification="biotypes",
                                biotypes=["low", "high"], conditions=["placebo", "mdma_high"],
                                stat=scipy.stats.spearmanr, CI_width=0.95, bootstrap_n_iter=1000, 
                                seed=9, verbose=True):
    
    cor_dfs = {}
    for condition in conditions:
        if verbose:
            print("-"*70 + f"\n{condition.upper().replace("_", " ")} CONDITION\n" + "-"*70)
        
        cor_rows = []
        
        if stratification == "biotypes":
            for biotype in biotypes:
                for var_y in vars_y:
                    df = speech_line_df[["condition", "biotype", var_y, var_x]].dropna()
                    df = df[(df["biotype"] == biotype) & (df["condition"] == condition)]
                    cor_row = compute_correlation(df, var_x, var_y, condition,
                                                  stat=stat, CI_width=CI_width, 
                                                  bootstrap_n_iter=bootstrap_n_iter,
                                                  seed=seed)
                    cor_row["biotype"] = biotype
                    cor_rows += [cor_row]
        
        elif stratification == None:
            for var_y in vars_y:
                    df = speech_line_df[["condition", var_y, var_x]].dropna()
                    df = df[(df["condition"] == condition)]
                    cor_row = compute_correlation(df, var_x, var_y, condition,
                                                  stat=stat, CI_width=CI_width, 
                                                  bootstrap_n_iter=bootstrap_n_iter,
                                                  seed=seed)
                    cor_rows += [cor_row]
        
        cor_df = pd.DataFrame(cor_rows)
        _, fdrs, _, _ = multipletests(cor_df["p"], alpha=0.05, method="fdr_bh")
        cor_df["FDR"] = fdrs
        cor_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        cor_dfs[condition] = cor_df
        
        if verbose:
            print(cor_df)
            print()

    return cor_dfs  


def compute_block_correlation(speech_line_df, var_x, vars_y, stratification="biotypes",
                                      biotypes=["low", "high"], conditions=["placebo", "mdma_high"],
                                      stat=scipy.stats.spearmanr, CI_width=0.95, bootstrap_n_iter=1000, 
                                      n_blocks=5, verbose=True):
    
    cor_dfs = {}
    for condition in conditions:
        if verbose:
            print("-"*70 + f"\n{condition.upper().replace("_", " ")} CONDITION\n" + "-"*70)
        
        cor_rows = []

        if stratification == "biotypes":
            for biotype in biotypes:
                for var_y in vars_y:
                    for block in range(1, n_blocks+1):
                        df = speech_line_df[["condition", "block", var_y, var_x]].dropna()
                        df = df[(df["condition"] == condition) & (df["block"] == block)]
                        cor_row = compute_correlation(df, var_x, var_y, condition, 
                                                      stat=stat, CI_width=CI_width, 
                                                      bootstrap_n_iter=bootstrap_n_iter)
                        cor_row["biotype"] = biotype
                        cor_row["block"] = block
                        cor_rows += [cor_row]

        elif stratification == None:
            for var_y in vars_y:
                for block in range(1, n_blocks+1):
                    df = speech_line_df[["condition", "block", var_y, var_x]].dropna()
                    df = df[(df["condition"] == condition) & (df["block"] == block)]
                    cor_row = compute_correlation(df, var_x, var_y, condition,
                                                  stat=stat, CI_width=CI_width, 
                                                  bootstrap_n_iter=bootstrap_n_iter)
                    cor_row["block"] = block
                    cor_rows += [cor_row]
                    
        cor_df = pd.DataFrame(cor_rows)
        _, fdrs, _, _ = multipletests(cor_df["p"], alpha=0.05, method="fdr_bh")
        cor_df["FDR"] = fdrs
        cor_df["sig"] = [p2sig(fdr) for fdr in fdrs]
        cor_dfs[condition] = cor_df
        
        if verbose:
            print(cor_df[["condition", "biotype", "var_x", "var_y", "n", "r", "CI_lower", "CI_upper", "FDR", "sig"]])
            print()

    return cor_dfs  


############################## EXPORTING ###############################

def export_stats_tables(stat_dicts, table_path, file_prefix,
                        conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):
    
    full_stat_table = pd.DataFrame()
    for condition in conditions:
        df = stat_dicts[condition].copy()
        stat_table = pd.DataFrame()
        stat_table["condition"] = [condition]*len(df)
        stat_table["var"] = df["var"].values
        if "blocks" in df.columns:
            stat_table["block"] = df["block"].values
        for biotype in biotypes:
            stat_table[f"n_{biotype}"] = df[f"n_{biotype}"].astype(int).values
            stat_table[f"med_[quantiles]_{biotype}"] = df[f"med_{biotype}"].apply(lambda x: "{:.2f} ".format(x)).values
            stat_table[f"med_[quantiles]_{biotype}"] = stat_table[f"med_[quantiles]_{biotype}"] + df[f"q1_{biotype}"].apply(lambda x: "[{:.2f}, ".format(x)).values
            stat_table[f"med_[quantiles]_{biotype}"] = stat_table[f"med_[quantiles]_{biotype}"] + df[f"q3_{biotype}"].apply(lambda x: "{:.2f}]".format(x)).values
        stat_table["U"] = df["U"].apply(lambda x: "{:2.1f}".format(x)).values
        stat_table["CLES"] = df["CLES"].apply(lambda x: "{:.2f}".format(x)).values
        stat_table["FDR"] = [sci_notation(x) if x < 0.01 else f"{x:.2f}" for x in df["FDR"]]
        stat_table["FDR"] = [f"{x}{sig}" for x, sig in zip(stat_table["FDR"], df["sig"])]
        full_stat_table = pd.concat([full_stat_table, stat_table])
    full_stat_table.to_csv(f"{table_path}/{file_prefix}_comp-biotypes.csv", index=None)
    
    full_stat_table = pd.DataFrame()
    for biotype in biotypes[::-1]:
        df = stat_dicts[biotype].copy()
        stat_table = pd.DataFrame()
        stat_table["biotype"] = [biotype]*len(df)
        stat_table["var"] = df["var"].values
        if "blocks" in df.columns:
            stat_table["block"] = df["block"].values
        for condition in conditions:
            condition = condition.split("_")[0]
            stat_table[f"n_{condition}"] = df[f"n_{condition}"].astype(int).values
            stat_table[f"med_[quantiles]_{condition}"] = df[f"med_{condition}"].apply(lambda x: "{:.2f} ".format(x)).values
            stat_table[f"med_[quantiles]_{condition}"] = stat_table[f"med_[quantiles]_{condition}"] + df[f"q1_{condition}"].apply(lambda x: "[{:.2f}, ".format(x)).values 
            stat_table[f"med_[quantiles]_{condition}"] = stat_table[f"med_[quantiles]_{condition}"] + df[f"q3_{condition}"].apply(lambda x: "{:.2f}]".format(x)).values
        stat_table["U"] = df["U"].apply(lambda x: "{:2.1f}".format(x)).values
        stat_table["CLES"] = df["CLES"].apply(lambda x: "{:.2f}".format(x)).values
        stat_table["FDR"] = [sci_notation(x) if x < 0.01 else f"{x:.2f}" for x in df["FDR"]]
        stat_table["FDR"] = [f"{x}{sig}" for x, sig in zip(stat_table["FDR"], df["sig"])]
        full_stat_table = pd.concat([full_stat_table, stat_table])
    full_stat_table.to_csv(f"{table_path}/{file_prefix}_comp-conditions.csv", index=None)
    

def export_multilevel_stats_tables(stat_dicts, var_type, table_path,
                                   conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):

    for dif_condition in conditions:
        for condition in conditions:
            for level, stat_dict in stat_dicts.items():
                df = stat_dict[dif_condition][condition].copy()
                stat_table = pd.DataFrame()
                stat_table["var"] = df["var"].values
                if level == "blocks":
                    stat_table["block"] = df["block"]
                for biotype in biotypes:
                    stat_table[f"n_{biotype}"] = df[f"n_{biotype}"].values
                    stat_table[f"med_[quantiles]_{biotype}"] = df[f"med_{biotype}"].apply(lambda x: "{:.2f} ".format(x)).values
                    stat_table[f"med_[quantiles]_{biotype}"] = stat_table[f"med_[quantiles]_{biotype}"] + df[f"q1_{biotype}"].apply(lambda x: "[{:.2f}, ".format(x)).values
                    stat_table[f"med_[quantiles]_{biotype}"] = stat_table[f"med_[quantiles]_{biotype}"] + df[f"q3_{biotype}"].apply(lambda x: "{:.2f}]".format(x)).values
                stat_table["U"] = df["U"].apply(lambda x: "{:,g}".format(x)).values
                stat_table["CLES"] = df["CLES"].apply(lambda x: "{:3.2f}".format(x)).values
                stat_table["FDR"] = [sci_notation(x) if x < 0.01 else f"{x:3.2f}" for x in df["FDR"]]
                stat_table["FDR"] = [f"{x}{sig}" for x, sig in zip(stat_table["FDR"], df["sig"])]
                stat_table.to_csv(f"{table_path}/{var_type}_{level}_dif-{dif_condition}_comp-{condition}.csv", index=None)
    
        for biotype in biotypes[::-1]:
            df = stat_dicts["sessions"][dif_condition][biotype].copy()
            stat_table = pd.DataFrame()
            stat_table["var"] = df["var"].values
            df.index = range(len(df))
            for condition in conditions:
                condition = condition.split("_")[0]
                stat_table[f"n_{condition}"] = df[f"n_{condition}"].values
                stat_table[f"med_[quantiles]_{condition}"] = df[f"med_{condition}"].apply(lambda x: "{:.2f} ".format(x)).values
                stat_table[f"med_[quantiles]_{condition}"] = stat_table[f"med_[quantiles]_{condition}"] + df[f"q1_{condition}"].apply(lambda x: "[{:.2f}, ".format(x)).values
                stat_table[f"med_[quantiles]_{condition}"] = stat_table[f"med_[quantiles]_{condition}"] + df[f"q3_{condition}"].apply(lambda x: "{:.2f}]".format(x)).values
            stat_table["U"] = df["U"].apply(lambda x: "{:,g}".format(x)).values
            stat_table["CLES"] = df["CLES"].apply(lambda x: "{:3.2f}".format(x)).values
            stat_table["FDR"] = [sci_notation(x) if x < 0.01 else f"{x:3.2f}" for x in df["FDR"]]
            stat_table["FDR"] = [f"{x}{sig}" for x, sig in zip(stat_table["FDR"], df["sig"])]
            stat_table.to_csv(f"{table_path}/{var_type}_sessions_dif-{dif_condition}_comp-{biotype}.csv", index=None)


def export_correlation_tables(stat_dicts, vars, var_type, table_path, 
                              level="sessions", stratification="biotype",
                              conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):
    
    for condition in conditions:

        if stratification == "biotype":
            stat_table = pd.DataFrame()
            df = stat_dicts[condition].copy()
            var_x = df["var_x"].values[0]
            vars = list(set(df["var_y"]))
            for var_y in vars:
                stat_row = pd.DataFrame()
                stat_row["var_x"] = [var_x]
                stat_row["var_y"] = [var_y]
                for biotype in biotypes:
                    biotype_df = df[(df["biotype"] == biotype) & (df["var_y"] == var_y)]
                    stat_row[f"n_{biotype}"] = biotype_df["n"].values
                    stat_row[f"r_[CI]_{biotype}"] = [f"{r:3.2f} [{CI_lower:3.2f}, {CI_upper:3.2f}]" for r, CI_lower, CI_upper in zip(biotype_df["r"], biotype_df["CI_lower"], biotype_df["CI_upper"])]
                    stat_row[f"FDR_{biotype}"] = [f"{sci_notation(x)}{sig}" if x < 0.01 else f"{x:3.2f}" for x, sig in zip(biotype_df["FDR"], biotype_df["sig"])]
                stat_table = pd.concat([stat_table, stat_row])

        elif stratification == None:
            df = stat_dicts[condition].copy()
            stat_table = df[["var_x", "var_y"]]
            if level == "blocks":
                stat_table["block"] = df["block"].values
            stat_table["n"] = df["n"].values
            stat_table["r_[CI]"] = [f"{r:3.2f} [{CI_lower:3.2f}, {CI_upper:3.2f}]" for r, CI_lower, CI_upper in zip(df["r"], df["CI_lower"], df["CI_upper"])]
            stat_table["FDR"] = [sci_notation(x) if x < 0.01 else f"{x:3.2f}" for x in df["FDR"]]
            stat_table["FDR"] = [f"{x}{sig}" for x, sig in zip(stat_table["FDR"], df["sig"])]
            
        stat_table.to_csv(f"{table_path}/{var_type}_{level}_cor-{condition}.csv", index=None)

