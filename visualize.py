import utils

import collections
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score


font_path = "style/cmu/cmunss.ttf"
font_manager.fontManager.addfont(font_path)

font_bold_path = "style/cmu/cmunsx.ttf"
font_manager.fontManager.addfont(font_bold_path)

palettes = {"low": ["#D5A265",
                    "#DE9E59", 
                    "#B89233", 
                    "#82651D", 
                    "#4D3D0F"],
            "high": ["#6E0A2B",
                     "#941932",
                     "#BD3026",
                     "#D0623F", 
                     "#DE7850"]}

condition2title = {"placebo": "Placebo", "mdma_high": "MDMA"}
biotype2title = {"low": r"NTN$_{\rm A–}$", "high": r"NTN$_{\rm A+}$"}
biotype2color = {"low": "#E6BE81", "high": "#FF3704"}
biotype2pointcolor = {"low": "#B79C71", "high": "#B4391E"}
biotype2linestyle = {"low": "--", "high": "-"}
biotype2alpha = {"low": 0.3, "high": 0.6}


########################################################################
############################## SPEECH DATA #############################
########################################################################

def plot_speech_biotype_differences(data_df, stat_dfs, vars, var2yrange, var2label, 
                                    figure_path, figure_suffix, 
                                    biotypes=["low", "high"], conditions=["placebo", "mdma_high"], 
                                    plot_points=True, plot_conditions=True, plot_biotypes=True,
                                    yn=0.15, ytn=2.5, bn=1.25, cn=5):

    fig, axes = plt.subplots(1, 3, figsize=(6.25, 0.9), squeeze=False)

    for var_i, var in enumerate(vars):

        max_y = data_df[var].max()
        
        for condition_i, condition in enumerate(conditions):
        
            condition_y = condition_i * cn
            condition_stat_df = stat_dfs[condition]
            df = data_df[data_df["condition"] == condition]
            for biotype_i, biotype in enumerate(biotypes):
                c = biotype2color[biotype]
                biotype2fill = {"high": c, "low": "none"}
                y = df[df["biotype"] == biotype][var].dropna()
                x = [condition_y+(biotype_i*bn)]*len(y)
                p = axes[0,var_i].violinplot(y, positions=[x[0]], widths=[2.5], 
                                             side=biotype)
                for pc in p["bodies"]:
                    pc.set_facecolor(c)
                    pc.set_alpha(0.35)
                    pc.set_edgecolor(c)
                for partname in ("cbars", "cmins", "cmaxes"):
                    vp = p[partname]
                    vp.set_linewidth(0)
                y_med = condition_stat_df.loc[var, f"med_{biotype}"]
                axes[0,var_i].scatter([x[0]], [y_med], 
                                      color=c, alpha=0.9, s=22, linewidth=1.25, zorder=100, 
                                      facecolors=biotype2fill[biotype])
                axes[0,var_i].plot(x[:2], [y.quantile(0.25), y.quantile(0.25)+y.std()], 
                                   color=c, alpha=0.9, linewidth=1.5, clip_on=False)

                if plot_points:
                    biotype_df = df[df["biotype"] == biotype]
                    axes[0,var_i].scatter([x[0]]*len(biotype_df), biotype_df[var], 
                                          color=biotype2pointcolor[biotype], 
                                          edgecolor="none", alpha=0.9, s=5, 
                                          linewidth=0, zorder=200)

        n_sig = 0
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
            max_y = data_df[var].dropna().max()
            star_y = max_y + (var2yrange[var][-1]*0.075)
            condition_stat_df = stat_dfs[condition]
            fdr = condition_stat_df.loc[var, "FDR"]
            if fdr < 0.05:
                axes[0,var_i].text(condition_y+(bn*0.5), star_y, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
                axes[0,var_i].plot([condition_y, condition_y+bn], [star_y]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                n_sig += 1

        for biotype_i, biotype in enumerate(biotypes):
            x = [(biotype_i*bn)]*len(y)
            max_y = df[var].dropna().max()
            max_y = data_df[~((data_df["condition"] == "placebo") & (data_df["biotype"] == "low"))][var].max()
            biotype_stat_df = stat_dfs[biotype]
            fdr = biotype_stat_df.loc[var, "FDR"]
            star_y = max_y + (var2yrange[var][-1]*0.075) + (n_sig*(var2yrange[var][-1]*0.175))
            if fdr < 0.05:
                mid_x = x[0] + (x[0]+cn - x[0])*0.5
                max_sig_y = max_y + (n_sig*(max_y*0.25)) 
                if biotype == "high":
                    max_sig_y = df[~((df["condition"] == "placebo") & (df["biotype"] == "low"))][var].max()
                    max_sig_y = max_sig_y + (max_y*0.15)
                axes[0,var_i].plot([x[0], x[0]+cn], [star_y]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[0,var_i].text(mid_x, star_y, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
        
        axes[0,var_i].set_xlim([-2.5, cn+2.5])
        axes[0,var_i].set_xticks([])

        ylabels = [var2yrange[var][0], var2yrange[var][-1]/2, var2yrange[var][-1]]
        axes[0,var_i].set_ylim(var2yrange[var])
        axes[0,var_i].set_yticks(ylabels)
        axes[0,var_i].set_yticklabels([str(int(label)) for label in ylabels], 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
        
        axes[0,var_i].set_ylabel(var2label[var], 
                                 font=font_manager.FontProperties(fname=font_path, size=10))
        
        axes[0,var_i].spines["top"].set_visible(False)
        axes[0,var_i].spines["bottom"].set_visible(False)
        axes[0,var_i].spines["right"].set_visible(False)

        if len(vars) < 3:
            for dir in ["top", "bottom", "right", "left"]:
                axes[0,-1].spines[dir].set_visible(False)
                axes[0,-1].set_xticks([])
                axes[0,-1].set_yticks([])
    
        max_y = ylabels[-1]
        y_line = max_y+(yn)+(n_sig*yn*ytn)+(max_y*0.35)
        y_text = max_y+yn+(n_sig*yn*ytn)+(max_y*0.415)
        if plot_conditions:
            axes[0,var_i].plot([-2, 2.8], [[y_line]*2, [y_line]*2], 
                               c="black", linewidth=0.75, clip_on=False)
            axes[0,var_i].plot([3.5, 8.3], [[y_line]*2, [y_line]*2], 
                               c="black", linewidth=0.75, clip_on=False)
            axes[0,var_i].text(0.4, y_text, "Placebo", ha="center", 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,var_i].text(5.9, y_text, "MDMA", ha="center", 
                               font=font_manager.FontProperties(fname=font_path, size=10))

        if plot_biotypes:
            axes[0,var_i].text(-3.5, -0.1, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,var_i].text(-1.5, -0.1, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,var_i].text(2, -0.1, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,var_i].text(4, -0.1, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
        
    plt.subplots_adjust(hspace=0.875, wspace=0.85)
    plt.savefig(f"{figure_path}/speech_{figure_suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()


########################################################################
#################### LOGISTIC REGRESSION CLASSIFIERS ###################
########################################################################

def plot_classifier_comparison(nlp_clf_metrics, gpt_clf_metrics, figure_path):

    fig, axes = plt.subplots(1, 4, figsize=(5.75, 1), squeeze=False)
    
    categories = ["NLP", "GPT-5"]
    
    i = 0
    
    axes[0,0].set_ylabel("Accuracy", 
                         font=font_manager.FontProperties(fname=font_path, size=12), labelpad=12)
    
    for split in ["valid", "test"]:
        for condition in ["placebo", "mdma_high"]:
        
    
            axes[0,i].plot([-0.65, 1.65], [50, 50], color="gray", linestyle="--", linewidth=0.65)
            
            val_nlp = nlp_clf_metrics[condition][f"accuracy_{split}"]*100
            val_gpt = gpt_clf_metrics[condition][f"accuracy_{split}"]*100
            axes[0,i].bar(categories, [val_nlp, val_gpt], color="gray", alpha=0.3)
    
            for cat_i, metrics in zip([0, 1], [nlp_clf_metrics, gpt_clf_metrics]):
                axes[0,i].plot([cat_i, cat_i], [metrics[condition][f"CI_lower_{split}"]*100, 
                                                metrics[condition][f"CI_upper_{split}"]*100], 
                               c="black", linewidth=1)
    
            axes[0,i].set_title(condition2title[condition], 
                                font=font_manager.FontProperties(fname=font_path, size=11), pad=15)
            axes[0,i].plot([-0.65, 1.65], [120, 120], 
                           color="black", linewidth=0.75, clip_on=False)
            
            axes[0,i].set_xlim([-0.65, 1.65])
            axes[0,i].set_xticklabels(categories, rotation=60, ha="center", 
                                      font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,i].set_ylim([0, 100])
            axes[0,i].set_yticks([0, 50, 100])
            axes[0,i].set_yticklabels(["0%", "50%", "100%"], 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
            
            axes[0,i].spines["top"].set_visible(False)
            axes[0,i].spines["right"].set_visible(False)
            
            i += 1
    
    plt.subplots_adjust(hspace=0.965, wspace=0.65)
    plt.savefig(f"{figure_path}/classifier_comparison.png", 
                dpi=300, bbox_inches="tight")
    plt.show()


def plot_roc_auc(gpt_clfs, speech_line_df, figure_path, 
                 splits=["valid", "test"], conditions=["placebo", "mdma_high"], 
                 condition2line_thres={"placebo": 20, "mdma_high": 20},
                 x_min=2, x_max=30):
    
    condition2title = {"placebo": "Placebo", "mdma_high": "MDMA"}
    x_ticks = range(0, x_max+5, 5)
    ticks = [0, 0.5, 1]

    df = speech_line_df.copy()
    df = df[df["split"].isin(splits)]
    df["index"] = df.index
    session2lines = {}
    for session in set(speech_line_df["session"]):
        session2lines[session] = len(df[df["session"] == session])
    df["n_lines"] = [session2lines[session] for session in df["session"]]
    df = df.sort_values(["n_lines", "index"])
    
    fig, axes = plt.subplots(2, len(conditions), figsize=(7.75, 3))
    
    for i, condition in enumerate(conditions):
        condition_df = df[df["condition"] == condition]
        condition_sessions = collections.OrderedDict({s: a for s, a in zip(condition_df["session"], 
                                                                           condition_df["n_lines"])}).keys() 
        
        clf = gpt_clfs[condition]
        clf_vars = clf.feature_names_in_
    
        x_thres = condition2line_thres[condition]

        axes[0,i].plot([0,1], [0,1], c="gray", linewidth=0.75, linestyle="--", alpha=1)
        
        for x_i in range(x_min, x_max+1):
            interval_sessions_df = pd.DataFrame()
            for session in condition_sessions:
                session_lines_df = df[df["session"] == session]
                if len(session_lines_df) >= x_i:
                    interval_lines_df = session_lines_df.head(x_i)
                    interval_sessions_df = pd.concat([interval_sessions_df, interval_lines_df])
    
            if len(interval_sessions_df) > 0:
                x = interval_sessions_df[clf_vars]
                y_true = np.array([0 if biotype == "low" else 1 for biotype in interval_sessions_df["biotype"]])
                if len(set(y_true)) > 1:
                    y_probs = clf.predict_proba(x)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_probs)
                    auc_score = roc_auc_score(y_true=y_true, y_score=y_probs)

                    axes[0,i].plot(fpr, tpr, c="gray", linewidth=0.75, alpha=0.25)
                    if x_i == x_thres:
                        axes[0,i].plot(fpr, tpr, c="black", linewidth=1, alpha=1, zorder=1000)
            
                    axes[1,i].scatter(x_i, auc_score, c="gray", s=15, alpha=0.5)
                    if x_i == x_thres:
                        axes[1,i].scatter(x_i, auc_score, c="black", s=20, alpha=1)
                        axes[1,i].text(x_i, auc_score+0.125, f"AUC = {auc_score:3.2f}", ha="center", 
                                       font=font_manager.FontProperties(fname=font_path, size=9))
    
        axes[0,i].plot([-0.1, 1.1], [1.6, 1.6], c="black", linewidth=0.75, clip_on=False)
        
        axes[0,i].set_title(condition2title[condition], 
                            font=font_manager.FontProperties(fname=font_path, size=12), pad=30)
        axes[0,i].set_xlim([ticks[0]-0.1, ticks[-1]+0.1])
        axes[0,i].set_xticks(ticks)
        axes[0,i].set_xticklabels(ticks, font=font_manager.FontProperties(fname=font_path, size=9))
        axes[0,i].set_ylim([ticks[0]-0.1, ticks[-1]+0.1])
        axes[0,i].set_yticks(ticks)
        axes[0,i].set_yticklabels(ticks, font=font_manager.FontProperties(fname=font_path, size=9))
        axes[0,i].spines["top"].set_visible(False)
        axes[0,i].spines["right"].set_visible(False)
        axes[0,i].set_xlabel("False positive rate", 
                             font=font_manager.FontProperties(fname=font_path, size=10), labelpad=5)
        axes[0,i].set_ylabel("True positive rate", 
                             font=font_manager.FontProperties(fname=font_path, size=10), labelpad=5)
    
        axes[1,i].set_xlim([x_min, x_max])
        axes[1,i].set_xticks(x_ticks)
        axes[1,i].set_xticklabels(x_ticks, font=font_manager.FontProperties(fname=font_path, size=9))
        axes[1,i].set_ylim([ticks[0]-0.1, ticks[-1]+0.1])
        axes[1,i].set_yticks(ticks)
        axes[1,i].set_yticklabels(ticks, font=font_manager.FontProperties(fname=font_path, size=9))
        axes[1,i].spines["top"].set_visible(False)
        axes[1,i].spines["right"].set_visible(False)
        axes[1,i].set_xlabel("Number of utterances", 
                             font=font_manager.FontProperties(fname=font_path, size=10), labelpad=5)
        axes[1,i].set_ylabel("ROC-AUC", 
                             font=font_manager.FontProperties(fname=font_path, size=10), labelpad=5)
    
    plt.subplots_adjust(hspace=0.8, wspace=0.375)
    plt.savefig(f"{figure_path}/roc_auc.png", 
                dpi=300, bbox_inches="tight")
    plt.show()
    print()


def plot_accuracy_by_lines(gpt_clfs, speech_line_df, figure_path, 
                           splits=["valid", "test"], conditions=["placebo", "mdma_high"], 
                           condition2line_thres={"placebo": 20, "mdma_high": 20},
                           x_min=1, x_max=70, bootstrap_n_iter=1000, CI_width=0.95):
    
    x_ticks = range(0, x_max+10, 10)
    y_ticks = [0, 0.5, 1]
    y_tick_labels = ["0%", "50%", "100%"]
    
    df = speech_line_df.copy()
    df = df[df["split"].isin(splits)]
    df["index"] = df.index
    session2lines = {}
    for session in set(speech_line_df["session"]):
        session2lines[session] = len(df[df["session"] == session])
    df["n_lines"] = [session2lines[session] for session in df["session"]]
    df = df.sort_values(["n_lines", "index"])
        
    for condition in conditions:
        print("-"*70 + f"\n{condition.upper()}\n" + "-"*70)
        condition_df = df[df["condition"] == condition]
        condition_sessions = collections.OrderedDict({s: a for s, a in zip(condition_df["session"], 
                                                                           condition_df["n_lines"])}).keys() 
        
        fig, axes = plt.subplots(len(condition_sessions), 1, figsize=(3.12, len(condition_sessions)*0.85))
    
        clf = gpt_clfs[condition]
        clf_vars = clf.feature_names_in_
    
        x_thres = condition2line_thres[condition]
        
        for i, session in enumerate(condition_sessions):
            session_lines_df = df[df["session"] == session]
            x_session, y_session, y_CI_lower, y_CI_upper = [], [], [], []
            for x in range(x_min, x_max+1):
                if len(session_lines_df) >= x:
                    interval_lines_df = session_lines_df.head(x)
                    biotype = interval_lines_df["biotype"].values[0]
                    y_true = [0 if biotype == "low" else 1 for i in range(len(interval_lines_df))]
                    y_pred = clf.predict(interval_lines_df[clf_vars])
                    y = accuracy_score(y_true, y_pred)
                    x_session += [x]
                    y_session += [y]
    
                    interval_lines_df.index = range(len(interval_lines_df))
                    y_boot = []
                    for boot_i in range(bootstrap_n_iter):
                        boot = np.random.choice(len(interval_lines_df), 
                                                size=len(interval_lines_df), replace=True)
                        y_pred = clf.predict(interval_lines_df.iloc[boot][clf_vars])
                        y_boot += [accuracy_score(y_true, y_pred)]
                    y_CI_lower += [sorted(y_boot)[int((1.0-CI_width)*len(y_boot))]]
                    y_CI_upper += [sorted(y_boot)[int(CI_width*len(y_boot))]]
    
            axes[i].plot(x_ticks, [0.5]*len(x_ticks), 
                         linewidth=0.75, linestyle="--", c="gray")
            
            mask = np.array(y_session) >= 0.5
            axes[i].fill_between(x_session, y_CI_lower, y_CI_upper, 
                                 color="gray", alpha=0.2)
            axes[i].fill_between(np.where(mask, x_session, np.nan), 
                                 np.where(mask, y_CI_lower, np.nan), 
                                 np.where(mask, y_CI_upper, np.nan),
                                 color="green", alpha=0.1)
    
            axes[i].plot(x_session, y_session, c="gray", alpha=0.5, linewidth=1.5)
            axes[i].plot(np.where(mask, x_session, np.nan), np.where(mask, y_session, np.nan), 
                         color="green", alpha=0.5, linewidth=1.5)
    
            axes[i].plot([x_thres, x_thres], [-0.15, 1.15], linewidth=1, c="black")
    
            axes[i].text(x_max+6, 0.45, interval_lines_df["biotype"][0].title(), ha="center", 
                         font=font_manager.FontProperties(fname=font_path, size=6.5))
            axes[i].text(x_max+12, 0.45, interval_lines_df["split"][0].title(), ha="center", 
                         font=font_manager.FontProperties(fname=font_path, size=6.5))
    
            axes[i].set_xlim([x_min, x_max])
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels([], 
                                    font=font_manager.FontProperties(fname=font_path, size=9))
            axes[i].set_ylim([-0.15, 1.15])
            axes[i].set_yticks(y_ticks)
            axes[i].set_yticklabels(y_tick_labels, 
                                    font=font_manager.FontProperties(fname=font_path, size=9))
            axes[i].spines["top"].set_visible(False)
            axes[i].spines["right"].set_visible(False)
    
        axes[i].set_xticklabels(x_ticks)
    
        axes[i].set_xlabel("Number of utterances", 
                           font=font_manager.FontProperties(fname=font_path, size=10), labelpad=5)
        fig.text(-0.04, 0.5, "Accuracy", 
                 ha="center", va="center", rotation=90, 
                 font=font_manager.FontProperties(fname=font_path, size=10))
    
        plt.subplots_adjust(hspace=0.275, wspace=0)
        plt.savefig(f"{figure_path}/accuracy_by_lines_{condition}.png", 
                    dpi=300, bbox_inches="tight")
        plt.show()
        print()


########################################################################
########################## BIOTYPE DIFFERENCES #########################
########################################################################

nlp2title = {
                 "text_predicted_emotion": "Valence in text",
                 "n_first_person_singular_to_n_words": "1st person singular",
                 "n_first_person_plural_to_n_words": "1st person plural",
                 "n_second_person_singular_to_n_words": "2nd person singular",
                 "n_third_person_singular_to_n_words": "3rd person singular",
                 "n_third_person_plural_to_n_words": "3rd person plural",
                 "n_past_to_n_words": "Past tense",
                 "n_present_to_n_words": "Present tense",
                 "n_future_to_n_words": "Future tense",
                 "n_hedges_to_n_words": "Hedging terms",
                 "n_words": "Word count",
                 "audio_predicted_emotion": "Valence in audio",
                 "words_per_sec": "Words per second",
                 "pause_duration_per_sec": "Pause duration per second",
                 "jitter": "Jitter",
                 "shimmer": "Shimmer",
                 "pitch_std": "Pitch standard deviation"
            }

nlp2label = {
                 "text_predicted_emotion": "Text valence",
                 "n_first_person_singular_to_n_words": "1st singular",
                 "n_first_person_plural_to_n_words": "1st plural",
                 "n_second_person_singular_to_n_words": "2nd singular",
                 "n_third_person_singular_to_n_words": "3rd singular",
                 "n_third_person_plural_to_n_words": "3rd plural",
                 "n_past_to_n_words": "Past",
                 "n_present_to_n_words": "Present",
                 "n_future_to_n_words": "Future",
                 "n_hedges_to_n_words": "Hedging",
                 "n_words": "Word count",
                 "audio_predicted_emotion": "Audio valence",
                 "words_per_sec": "Words/sec",
                 "pause_duration_per_sec": "Pause dur/sec",
                 "jitter": "Jitter",
                 "shimmer": "Shimmer",
                 "pitch_std": "Pitch SD"
            }

nlp2description = {
                     "text_predicted_emotion": "Affective valence predicted in texts by RoBERTa fine-tuned on SEND",
                     "n_first_person_singular_to_n_words": "Number of first person singular words relative to word count",
                     "n_first_person_plural_to_n_words": "Number of first person plural words relative to word count",
                     "n_second_person_singular_to_n_words": "Number of second person singular words relative to word count",
                     "n_third_person_singular_to_n_words": "Number of third person singular words relative to word count",
                     "n_third_person_plural_to_n_words": "Number of third person plural words relative to word count",
                     "n_past_to_n_words": "Number of past tenese words relative to word count",
                     "n_present_to_n_words": "Number of present tense words relative to word count",
                     "n_future_to_n_words": "Number of future tense words relative to word count",
                     "n_hedges_to_n_words": "Number of hedging terms relative to word count",
                     "n_words": "Word count for the utterance",
                     "audio_predicted_emotion": "Affective valence predicted in audio by Whisper fine-tuned on SEND",
                     "words_per_sec": "Number of words per second",
                     "pause_duration_per_sec": "Duration of pauses per second",
                     "jitter": "Period-to-period variability of the pitch",
                     "shimmer": "Perturbation of the amplitude",
                     "pitch_std": "Standard deviation of the pitch"
                  }

nlp2yticks = {
                "text_predicted_emotion": [0, 0.5, 1],
                "n_first_person_singular_to_n_words": [0, 0.15, 0.3],
                "n_first_person_plural_to_n_words": [0, 0.15, 0.3],
                "n_second_person_singular_to_n_words": [0, 0.15, 0.3],
                "n_third_person_singular_to_n_words": [0, 0.15, 0.3],
                "n_third_person_plural_to_n_words": [0, 0.15, 0.3],
                "n_past_to_n_words": [0, 0.15, 0.3],
                "n_present_to_n_words": [0, 0.15, 0.3],
                "n_future_to_n_words": [0, 0.15, 0.3],
                "n_hedges_to_n_words": [0, 0.15, 0.3],
                "n_words": [0, 0.15, 0.3],
                "audio_predicted_emotion": [0, 0.5, 1],
                "words_per_sec": [0, 3, 6],
                "pause_duration_per_sec": [0, 0.5, 1],
                "jitter": [0, 0.05, 0.1],
                "shimmer": [0, 0.05, 0.1],
                "pitch_std": [0, 125, 250]
              }

def plot_feature_differences(filt_dfs, stat_dicts, speech_line_df, figure_path, file_prefix,
                             var2title, var2description, var2label, var2yticks, 
                             conditions=["placebo", "mdma_high"], biotypes=["low", "high"], n_blocks=5,
                             yn=0.125, ytn=2.25, bn=0.75, cn=5.5):
    
    for dif_condition in conditions:
        print("-"*85 + f"\n{dif_condition.upper()} CONDITION\n" + "-"*85)
        for dif_biotype in biotypes[::-1]:
            print("-"*70 + f"\nFEATURES DIFFERENTIATING THE {dif_biotype.upper()} BIOTYPE IN THE {dif_condition.upper()} CONDITION\n" + "-"*70)
            
            vars = list(filt_dfs[dif_condition][dif_biotype].columns)
            df = speech_line_df[["condition", "biotype", "block"] + vars]
            
            fig, axes = plt.subplots(len(vars), 3, 
                                     figsize=(4.7, (len(vars)*1.15)-0.55), squeeze=False)
            
            for var_i, var in enumerate(vars):
    
                max_y = df[var].dropna().max()
                y_range = var2yticks[var][-1]-var2yticks[var][0]
                
                n_sig = 0
                for condition_i, condition_y, condition in zip([0, 1], [0, cn], conditions):
    
                    for biotype_i, biotype in zip([-0.1, 0.1], biotypes):
                        for block in range(1, n_blocks+1):
                            block_df = df[(df["biotype"] == biotype) & (df["condition"] == condition) & (df["block"] == block)]
                            vals = block_df[var]
                            c = palettes[dif_biotype][var_i]
                            biotype2fill = {"high": c, "low": "none"}
                            y = block_df[block_df["biotype"] == biotype][var].dropna()
                            x = [block+biotype_i]*len(y)
                            p = axes[var_i,condition_i].violinplot(y, 
                                                                   positions=[x[0]], widths=[0.5], side=biotype)
                            for pc in p["bodies"]:
                                pc.set_facecolor(c)
                                pc.set_alpha(biotype2alpha[biotype]-0.1)
                                pc.set_edgecolor(c)
                            for partname in ("cbars", "cmins", "cmaxes"):
                                vp = p[partname]
                                vp.set_linewidth(0)
                            axes[var_i,condition_i].plot(x[:2], [y.quantile(0.25), y.quantile(0.25)+y.std()], 
                                                         color=c, alpha=biotype2alpha[biotype]+0.3, linewidth=1, clip_on=False)
                    
                    block_stat_df = stat_dicts["blocks"][dif_condition][condition]
                    block_maxes = {block: 0 for block in range(1, 6)}
                    for biotype_i, biotype in zip([-0.1, 0.1], biotypes):
                        for block in range(1,6):
                            block_dict = block_stat_df[(block_stat_df["block"] == block) & (block_stat_df["var"] == var)].squeeze().to_dict()
                            block_max = block_dict[f"max_{biotype}"]
                            if block_max > block_maxes[block]: 
                                block_maxes[block] = block_max
    
                        plot_blocks = block_stat_df[block_stat_df["var"] == var]["block"]
                        plot_blocks = [biotype_i + b for b in plot_blocks]
                        plot_meds = block_stat_df[block_stat_df["var"] == var][f"med_{biotype}"]
                        axes[var_i,condition_i].plot(plot_blocks, plot_meds, 
                                                     color=palettes[dif_biotype][var_i], alpha=1, linewidth=1, 
                                                     linestyle=biotype2linestyle[biotype])
                        
                    for block in range(1, n_blocks+1):
                        block_dict = block_stat_df[(block_stat_df["block"] == block) & (block_stat_df["var"] == var)].squeeze().to_dict()
                        fdr = block_dict["FDR"]
                        if fdr < 0.05:
                            y_star = block_maxes[block]+(y_range*0.05)
                            sig = utils.p2sig(fdr).replace("***", "*\n*\n*").replace("**", "*\n*")
                            axes[var_i,condition_i].text(block, y_star, sig, clip_on=False, 
                                                         ha="center", linespacing=0.4,
                                                         font=font_manager.FontProperties(fname=font_path, size=9))
    
                    condition_df = df[df["condition"] == condition]
                    for biotype_i, biotype in enumerate(biotypes):
                        c = palettes[dif_biotype][var_i]
                        biotype2fill = {"high": c, "low": "none"}
                        y = condition_df[condition_df["biotype"] == biotype][var].dropna()
                        x = [condition_y+(biotype_i*bn)]*len(y)
                        p = axes[var_i,2].violinplot(y, positions=[x[0]], widths=[2.5], side=biotype)
                        for pc in p["bodies"]:
                            pc.set_facecolor(c)
                            pc.set_alpha(biotype2alpha[biotype]-0.1)
                            pc.set_edgecolor(c)
                        for partname in ("cbars", "cmins", "cmaxes"):
                            vp = p[partname]
                            vp.set_linewidth(0)
                        axes[var_i,2].scatter([x[0]], [np.median(y)], 
                                              color=c, alpha=1, s=7, linewidth=1, zorder=100, 
                                              facecolors=biotype2fill[biotype])
                        axes[var_i,2].plot(x[:2], [y.quantile(0.25), y.quantile(0.25)+y.std()], 
                                           color=c, alpha=biotype2alpha[biotype]+0.3, linewidth=1, clip_on=False)
                        
                    condition_stat_df = stat_dicts["sessions"][dif_condition][condition]
                    fdr = condition_stat_df.loc[var, "FDR"]
                    if fdr < 0.05:
                        condition_max_y = df[var].max()
                        y_range = var2yticks[var][-1]-var2yticks[var][0]
                        y_star = condition_max_y+(y_range*0.15)
                        axes[var_i,2].text(condition_y+(bn*0.5), y_star, utils.p2sig(fdr), 
                                           clip_on=False, ha="center",
                                         font=font_manager.FontProperties(fname=font_path, size=9))
                        axes[var_i,2].plot([condition_y, condition_y+bn], [y_star]*2, 
                                           c="black", linewidth=0.75, clip_on=False)
    
                    if condition == conditions[1]:
                        n_sig += 1
                        
                for biotype_i, biotype in enumerate(biotypes):
                    x = [(biotype_i*bn)]*len(y)
                    biotype_stat_df = stat_dicts["sessions"][dif_condition][biotype]
                    fdr = biotype_stat_df.loc[var, "FDR"]
                    if fdr < 0.05:
                        mid_x = x[0] + (x[0]+cn - x[0])*0.5
                        y_range = var2yticks[var][-1]-var2yticks[var][0]
                        y_star = max_y+(y_range*0.15)+(n_sig*(y_range*0.3))
                        axes[var_i,2].plot([x[0], x[0]+cn], [y_star]*2, c="black", 
                                           linewidth=0.75, clip_on=False)
                        axes[var_i,2].text(mid_x, y_star, utils.p2sig(fdr), 
                                           clip_on=False, ha="center", 
                                           font=font_manager.FontProperties(fname=font_path, size=9))
                        n_sig += 1
    
                y_min = var2yticks[var][0]
                y_max = var2yticks[var][-1]
                
                title = var2title[var]
                axes[var_i,2].text(cn*1.75, y_max-((y_max-y_min)/5), title, c=c,
                                 font=font_manager.FontProperties(fname=font_bold_path, size=9.5))
        
                text = var2description[var]
                text = "\n".join(textwrap.wrap(text, width=45))
                axes[var_i,2].text(cn*1.75, y_max-((y_max-y_min)/2.5), text, c="black", va="top",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
    
                axes[var_i,2].set_xlim([-2.5, cn+2.25])
                axes[var_i,2].set_xticks([])
    
                axes[var_i,2].set_ylim([y_min-((y_max-y_min)/10), y_max+((y_max-y_min)/10)])
                axes[var_i,2].set_yticks(var2yticks[var])
                axes[var_i,2].set_yticklabels(var2yticks[var], 
                                              font=font_manager.FontProperties(fname=font_path, size=9))
                
                axes[var_i,2].spines["bottom"].set_visible(False)
                
                for plot_i in range(2):
                    axes[var_i,plot_i].set_xlim([0.5, 5.5])
                    axes[var_i,plot_i].set_xticks([1, 2, 3, 4, 5])
                    axes[var_i,plot_i].set_xticklabels([], font=font_manager.FontProperties(fname=font_path, size=9))
                    
                    axes[var_i,plot_i].set_ylim([y_min-((y_max-y_min)/10), y_max+((y_max-y_min)/10)])
                    axes[var_i,plot_i].set_yticks(var2yticks[var])
                    axes[var_i,plot_i].set_yticklabels(var2yticks[var], 
                                                       font=font_manager.FontProperties(fname=font_path, size=9))
    
                for plot_i in range(3):
                    axes[var_i,plot_i].spines["top"].set_visible(False)
                    axes[var_i,plot_i].spines["right"].set_visible(False)
    
                axes[var_i,0].set_ylabel(var2label[var], 
                                         font=font_manager.FontProperties(fname=font_path, size=9))
                axes[var_i,0].yaxis.set_label_coords(-0.375, 0.5)
    
                if var_i == 0:
                    if y_star > y_max:
                        y_max = y_star
                    y_line = y_max+(y_range*0.5)
                    y_text = y_max+(y_range*0.6)
                    
                    axes[0,0].plot([1, 5], [[y_line]*2, [y_line]*2], 
                                   c="black", linewidth=0.75, clip_on=False)
                    axes[0,0].text(3, y_text, condition2title[conditions[0]], 
                                   ha="center", font=font_manager.FontProperties(fname=font_path, size=10))
                        
                    axes[0,1].plot([1, 5], [[y_line]*2, [y_line]*2], 
                                   c="black", linewidth=0.75, clip_on=False)
                    axes[0,1].text(3, y_text, condition2title[conditions[1]], 
                                   ha="center", font=font_manager.FontProperties(fname=font_path, size=10))
      
                    axes[0,2].plot([-2, 2.8], [[y_line]*2, [y_line]*2], 
                                   c="black", linewidth=0.75, clip_on=False)
                    axes[0,2].plot([3.5, 8.3], [[y_line]*2, [y_line]*2], 
                                   c="black", linewidth=0.75, clip_on=False)
                    axes[0,2].text(0.4, y_text, condition2title[conditions[0]], 
                                   ha="center", font=font_manager.FontProperties(fname=font_path, size=10))
                    axes[0,2].text(5.9, y_text, condition2title[conditions[1]], 
                                   ha="center", font=font_manager.FontProperties(fname=font_path, size=10))
            
                if var_i == len(vars)-1:
                    for plot_i in range(2):
                        axes[var_i,plot_i].set_xticklabels([1, 2, 3, 4, 5], font=font_manager.FontProperties(fname=font_path, size=9))
                        axes[var_i,plot_i].set_xlabel("Block", font=font_manager.FontProperties(fname=font_path, size=10))
                    
                    y_max = var2yticks[var][-1]
                    y_label = y_min-((y_max-y_min)/10)
                    axes[var_i,2].text(-3.5, y_label, r"NTN$_{\rm A–}$", 
                                       ha="left", va="top", rotation=60, 
                                       font=font_manager.FontProperties(fname=font_path, size=10))
                    axes[var_i,2].text(-1.5, y_label, r"NTN$_{\rm A+}$", 
                                       ha="left", va="top", rotation=60, 
                                       font=font_manager.FontProperties(fname=font_path, size=10))
                    axes[var_i,2].text(2, y_label, r"NTN$_{\rm A–}$", 
                                       ha="left", va="top", rotation=60, 
                                       font=font_manager.FontProperties(fname=font_path, size=10))
                    axes[var_i,2].text(4, y_label, r"NTN$_{\rm A+}$", 
                                       ha="left", va="top", rotation=60, 
                                       font=font_manager.FontProperties(fname=font_path, size=10))
            
            plt.subplots_adjust(hspace=0.875, wspace=0.4)
            plt.savefig(f"{figure_path}/{file_prefix}_{dif_condition}_{dif_biotype}_participant-split_lines.png", 
                        dpi=300, bbox_inches="tight")
            plt.show()



########################################################################
########################## GRAMMATICAL PERSON ##########################
########################################################################

person_var2lineyrange = {"n_first_person_singular_to_n_words": [-0.022, 0.17],
                         "n_third_person_to_n_words": [-0.0055, 0.055]}

person_var2lineyticks = {"n_first_person_singular_to_n_words": [0, 0.075, 0.15],
                         "n_third_person_to_n_words": [0, 0.025, 0.05]}

person_var2distyrange = {"n_first_person_singular_to_n_words": [-0.022, 0.32],
                         "n_third_person_to_n_words": [-0.015, 0.215]}

person_var2distyticks = {"n_first_person_singular_to_n_words": [0, 0.15, 0.3],
                         "n_third_person_to_n_words": [0, 0.1, 0.2]}

def plot_person_figure(speech_line_df, stat_dfs, cor_dfs, vars, gpt_var, figure_path,
                       var2lineyrange=person_var2lineyrange, 
                       var2lineyticks=person_var2lineyticks, 
                       var2distyrange=person_var2distyrange, 
                       var2distyticks=person_var2distyticks, 
                       ylabel="Degree of boundary openness rated by GPT",
                       suffix="boundary_openness",
                       conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                       yn=0.2, ytn=0.225, bn=0.75, cn=5.5):
    
    
    yn = yn * (var2lineyrange[vars[0]][1]-var2lineyrange[vars[0]][0])
    
    biotype2pcolor = {"low": "#E8CFA7", "high": "#F39881"}
    biotype2psize = {"low": 45, "high": 60}
    
    fig, axes = plt.subplots(len(vars), 5, figsize=(7.25, 2.5), 
                             gridspec_kw={'width_ratios': [1.35,1,1,1,1]}, squeeze=False)
    
    for var_i, var in enumerate(vars):
        
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
        
            max_y = speech_line_df[var].dropna().max()
            
            df = speech_line_df.copy()
            df = df[df["condition"] == condition]
            for biotype_i, biotype in enumerate(biotypes):
                c = biotype2color[biotype]
                biotype2fill = {"high": c, "low": "none"}
                y = df[df["biotype"] == biotype][var].dropna()
                x = [condition_y+(biotype_i*bn)]*len(y)
                p = axes[var_i,0].violinplot(y, positions=[x[0]], widths=[3], side=biotype)
                for pc in p["bodies"]:
                    pc.set_facecolor(c)
                    pc.set_alpha(0.4)
                    pc.set_edgecolor(c)
                for partname in ("cbars", "cmins", "cmaxes"):
                    vp = p[partname]
                    vp.set_linewidth(0)
                axes[var_i,0].scatter([x[0]], [np.median(y)], 
                                      color=c, alpha=1, s=12, linewidth=1.25, 
                                      facecolors=biotype2fill[biotype])
                axes[var_i,0].plot(x[:2], [y.quantile(0.25), y.quantile(0.75)], 
                                   color=c, alpha=0.9, linewidth=1.5, clip_on=False)
        
        n_sig = 0
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
            max_y = speech_line_df[speech_line_df["condition"] == condition][var].dropna().max()
            star_y = max_y+yn+(n_sig*yn*ytn) - ((var2distyrange[var][1]-var2distyrange[var][0])*0.075)
            condition_stat_df = stat_dfs[condition]
            fdr = condition_stat_df.loc[var, "FDR"]
            if fdr < 0.05:
                axes[var_i,0].plot([condition_y, condition_y+bn], [star_y]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[var_i,0].text(condition_y+(bn*0.5), star_y, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
        
        for biotype_i, biotype in enumerate(biotypes):
            max_y = df[var].dropna().max()
            y = df[df["biotype"] == biotype][var].dropna()
            x = [(biotype_i*bn)]*len(y)
            biotype_stat_df = stat_dfs[biotype]
            fdr = biotype_stat_df.loc[var, "FDR"]
            if fdr < 0.05:
                mid_x = x[0] + (x[0]+cn - x[0])*0.5
                star_y = max_y+(yn)+(n_sig*yn*ytn) + ((var2distyrange[var][1]-var2distyrange[var][0])*0.075)
                axes[var_i,0].plot([x[0], x[0]+cn], [star_y]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[var_i,0].text(mid_x, star_y, utils.p2sig(fdr), clip_on=False, ha="center",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
    
        axes[var_i,0].set_ylim(var2distyrange[var])
        axes[var_i,0].set_yticks(var2distyticks[var])
        axes[var_i,0].set_yticklabels(var2distyticks[var], 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
        axes[var_i,0].spines["top"].set_visible(False)
        axes[var_i,0].spines["right"].set_visible(False)
        axes[var_i,0].spines["bottom"].set_visible(False)
        axes[var_i,0].set_xlim([-3, 8])
        axes[var_i,0].set_xticks([])
        axes[var_i,0].set_xticklabels([])
    
        if var_i == len(vars)-1:
            label_y = -0.02
            axes[var_i,0].text(-3.5, label_y, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,0].text(-1.5, label_y, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,0].text(2, label_y, r"NTN$_{\rm A–}$", ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,0].text(4, label_y, r"NTN$_{\rm A+}$", ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
    
    title_y = max_y+yn+(n_sig*yn*ytn)+0.165
    pos_y = 0.03
    axes[0,0].plot([-2, 2.8], [[title_y]*2, [title_y]*2], 
                   c="black", linewidth=0.75, clip_on=False)
    axes[0,0].plot([3.5, 8.3], [[title_y]*2, [title_y]*2], 
                   c="black", linewidth=0.75, clip_on=False)
    axes[0,0].text(0.4, title_y+pos_y, "Placebo", ha="center", 
                   font=font_manager.FontProperties(fname=font_path, size=10))
    axes[0,0].text(5.9, title_y+pos_y, "MDMA", ha="center", 
                   font=font_manager.FontProperties(fname=font_path, size=10))
        
    axes[0,0].set_ylabel("1st person\nsingular", labelpad=3, 
                         font=font_manager.FontProperties(fname=font_path, size=10))
    axes[1,0].set_ylabel("3rd person\nsingular & plural", labelpad=6, 
                         font=font_manager.FontProperties(fname=font_path, size=10))
    
    fig.text(-0.0075, 0.5, r"$\it{N}$ person words to $\it{N}$ words", 
             va="center", rotation="vertical",
             font=font_manager.FontProperties(fname=font_path, size=10))
    axes[1,0].plot([-11.175, -11.175], [[0]*2, [0.55]*2], 
                   c="black", linewidth=0.75, clip_on=False)
    
    for row_i, var_j in enumerate(vars):
        col_i = 0
        for condition in conditions:
            for biotype_i, biotype in enumerate(biotypes):
    
                col_i += 1
                df = speech_line_df[speech_line_df["biotype"] == biotype]
                df = df[df["condition"] == condition]
                df = df[[gpt_var, var_j]].dropna()
                
                axes[row_i,col_i].scatter(df[gpt_var], df[var_j], 
                                          s=6, alpha=0.25, c=biotype2color[biotype])
        
                m, b = np.polyfit(df[gpt_var], df[var_j], 1)
                axes[row_i,col_i].plot(df[gpt_var], m*df[gpt_var]+b, 
                                       linewidth=0.75, color="black") 
        
                stat_df = cor_dfs[condition]
                stat_df = stat_df[stat_df["var_x"] == gpt_var]
                stat_df = stat_df[stat_df["var_y"] == var_j]
                stat_df = stat_df[stat_df["biotype"] == biotype]
                if len(stat_df) > 0:
                    r = stat_df["r"].values[0]
                    sig = stat_df["sig"].values[0]
                    axes[row_i,col_i].text(0.5, var2distyrange[var_j][-1], r"$\it{ρ}$" + f" = {r:3.2f}{sig}", 
                                           ha="center",
                                                 font=font_manager.FontProperties(fname=font_path, size=8))
    
                axes[row_i,col_i].set_xlim([-0.15, 1.15])
                axes[row_i,col_i].set_xticks([0, 0.5, 1])
                axes[row_i,col_i].set_xticklabels([])
                axes[row_i,col_i].set_ylim(var2distyrange[var_j])
                axes[row_i,col_i].set_yticks(var2distyticks[var_j])
                axes[row_i,col_i].set_yticklabels([])
                axes[row_i,col_i].spines["top"].set_visible(False)
                axes[row_i,col_i].spines["right"].set_visible(False)
                axes[row_i,col_i].set_xticklabels([0, 0.5, 1], 
                                                  font=font_manager.FontProperties(fname=font_path, size=9))
                axes[row_i,col_i].set_yticklabels(var2distyticks[var_j], 
                                                  font=font_manager.FontProperties(fname=font_path, size=9))  
    
    title_y = max_y+yn+(n_sig*yn*ytn)+0.165
    pos_y = 0.03
    axes[0,1].plot([0, 3], [[title_y]*2, [title_y]*2], 
                   c="black", linewidth=0.75, clip_on=False)
    axes[0,3].plot([0, 3], [[title_y]*2, [title_y]*2], 
                   c="black", linewidth=0.75, clip_on=False)
    axes[0,1].text(1.5, title_y+pos_y, "Placebo", ha="center", 
                   font=font_manager.FontProperties(fname=font_path, size=10))
    axes[0,3].text(1.5, title_y+pos_y, "MDMA", ha="center", 
                   font=font_manager.FontProperties(fname=font_path, size=10))
    
    fig.text(0.61, -0.05, ylabel, ha="center",
             font=font_manager.FontProperties(fname=font_path, size=10))
    
    plt.subplots_adjust(hspace=0.55, wspace=0.515)
    plt.savefig(f"{figure_path}/person_X_{suffix}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print()



########################################################################
########################### AFFECTIVE VALENCE ##########################
########################################################################

def plot_valence_differences(speech_line_df, valence_stat_dicts, valence_vars, figure_path,
                             conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                             n_blocks=5, yn=0.135, ytn=2.25, bn=0.75, cn=5.5):
    
    fig, axes = plt.subplots(len(valence_vars), 3, 
                             figsize=(6.35, (len(valence_vars)*1.1)-0.55), 
                             gridspec_kw={'width_ratios': [3, 3, 1.75]}, squeeze=False)
    
    for var_i, var in enumerate(valence_vars):
    
        max_y = speech_line_df[var].dropna().max()
        
        n_sig = 0
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
            
            block_stat_df = valence_stat_dicts["blocks"][condition]
            block_maxes = {block: 0 for block in range(1, 6)}
            for biotype_i, biotype in zip([-0.1, 0.1], biotypes):
                for block in range(1,6):
                    block_dict = block_stat_df[(block_stat_df["block"] == block) & (block_stat_df["var"] == var)]
                    block_max = block_dict[f"max_{biotype}"].values[0]
                    if block_max > block_maxes[block]: 
                        block_maxes[block] = block_max
    
                df = speech_line_df[speech_line_df["condition"] == condition]
                for block in range(1,6):
                    c = biotype2color[biotype]
                    biotype2fill = {"high": c, "low": "none"}
                    y = df[(df["biotype"] == biotype) & (df["block"] == block)][var].dropna()
                    x = block+(biotype_i*bn)
                    p = axes[var_i,condition_i].violinplot(y, positions=[x], widths=[0.5], 
                                                           side=biotype)
                    for pc in p["bodies"]:
                        pc.set_facecolor(c)
                        pc.set_alpha(0.4)
                        pc.set_edgecolor(c)
                    for partname in ("cbars", "cmins", "cmaxes"):
                        vp = p[partname]
                        vp.set_linewidth(0)
                    axes[var_i,condition_i].plot([x, x], [y.quantile(0.25), y.quantile(0.75)], 
                                                 color=c, alpha=0.9, linewidth=1.5, clip_on=False)
                
                plot_blocks = block_stat_df[block_stat_df["var"] == var]["block"]
                plot_blocks = [biotype_i + b for b in plot_blocks]
                plot_meds = block_stat_df[block_stat_df["var"] == var][f"med_{biotype}"]
                
                axes[var_i,condition_i].plot(plot_blocks, plot_meds, 
                                             color=biotype2color[biotype], alpha=0.8, linewidth=1, 
                                             linestyle=biotype2linestyle[biotype]) 
                            
            for block in range(1, n_blocks+1):
                block_dict = block_stat_df[(block_stat_df["block"] == block) & (block_stat_df["var"] == var)]
                fdr = block_dict["FDR"].values[0]
                if fdr < 0.05:
                    sig = utils.p2sig(fdr).replace("***", "*\n*\n*").replace("**", "*\n*")
                    axes[var_i,condition_i].text(block, block_maxes[block]+0.02, sig, 
                                                 clip_on=False, ha="center", linespacing=0.45,
                                                 font=font_manager.FontProperties(fname=font_path, size=9))
    
            df = speech_line_df[speech_line_df["condition"] == condition]
            for biotype_i, biotype in enumerate(biotypes):
                c = biotype2color[biotype]
                y = df[df["biotype"] == biotype][var].dropna()
                x = [condition_y+(biotype_i*bn)]*len(y)
                p = axes[var_i,2].violinplot(y, positions=[x[0]], widths=[3], side=biotype)
                for pc in p["bodies"]:
                    pc.set_facecolor(biotype2color[biotype])
                    pc.set_alpha(0.4)
                    pc.set_edgecolor(biotype2color[biotype])
                for partname in ("cbars", "cmins", "cmaxes"):
                    vp = p[partname]
                    vp.set_linewidth(0)
                axes[var_i,2].scatter([x[0]], [np.median(y)], 
                                      color=c, alpha=1, s=12, linewidth=1.25, 
                                      facecolors=biotype2fill[biotype])
                axes[var_i,2].plot(x[:2], [y.quantile(0.25), y.quantile(0.75)], 
                                   color=c, alpha=0.9, linewidth=1.5, clip_on=False)
        
            condition_stat_df = valence_stat_dicts["sessions"][condition]
            fdr = condition_stat_df.loc[var, "FDR"]
            if fdr < 0.05:
                axes[var_i,2].plot([condition_y, condition_y+bn], [max_y+(yn)+(n_sig*yn*ytn)]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[var_i,2].text(condition_y+(bn*0.5), max_y+yn+(n_sig*yn*ytn)-0.02, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                 font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
                
        for biotype_i, biotype in enumerate(biotypes):
            x = [(biotype_i*bn)]*len(y)
            biotype_stat_df = valence_stat_dicts["sessions"][biotype]
            fdr = biotype_stat_df.loc[var, "FDR"]
            if fdr < 0.05:
                mid_x = x[0] + (x[0]+cn - x[0])*0.5
                axes[var_i,2].plot([x[0], x[0]+cn], [max_y+(yn)+(n_sig*yn*ytn)]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[var_i,2].text(mid_x, max_y+yn+(n_sig*yn*ytn)-0.02, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                   font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
    
        for plot_i in range(2):
            axes[var_i,plot_i].set_xlim([0.5, 5.5])
            axes[var_i,plot_i].set_xticks([1, 2, 3, 4, 5])
            axes[var_i,plot_i].set_xticklabels([1, 2, 3, 4, 5], 
                                               font=font_manager.FontProperties(fname=font_path, size=9))
            axes[var_i,plot_i].set_xlabel("Block", 
                                          font=font_manager.FontProperties(fname=font_path, size=10))
    
        for plot_i in range(3):
            axes[var_i,plot_i].set_ylim([0, 1.1])
            axes[var_i,plot_i].set_yticks([0, 0.5, 1])
            axes[var_i,plot_i].set_yticklabels([0, 0.5, 1], 
                                               font=font_manager.FontProperties(fname=font_path, size=9))
                
        axes[var_i,2].set_xlim([-2.5, cn+2.5])
        axes[var_i,2].set_xticks([])
        axes[var_i,2].spines["bottom"].set_visible(False)
        
        for plot_i in range(3):
            axes[var_i,plot_i].spines["top"].set_visible(False)
            axes[var_i,plot_i].spines["right"].set_visible(False)
    
        if var_i == 0:
            plot2yrange = {0: 1.1, 1: 1.1, 2: 1.1}
            plot2ymax = {0: 1.1, 1: 1.1, 2: 1.1}
    
            title_y = plot2ymax[0]+plot2yrange[0]*0.475
            axes[0,0].plot([1, 5], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
            axes[0,0].text(3, title_y+plot2yrange[0]*0.1, "Placebo", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
    
            title_y = plot2ymax[1]+plot2yrange[1]*0.475
            axes[0,1].plot([1, 5], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
            axes[0,1].text(3, title_y+plot2yrange[1]*0.1, "MDMA", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
    
            title_y = plot2ymax[2]+plot2yrange[2]*0.475
            axes[0,2].plot([-2, 2.8], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
            axes[0,2].plot([3.5, 8.3], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
            axes[0,2].text(0.4, title_y+(plot2yrange[2]*0.1), "Placebo", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
            axes[0,2].text(5.9, title_y+(plot2yrange[2]*0.1), "MDMA", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
    
        if var_i == len(valence_vars)-1:
            axes[var_i,2].text(-3.5, 0.05, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(-1.5, 0.05, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(2, 0.05, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(4, 0.05, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
    
    fig.text(0.0475, 0.5, "Regression logit", va="center", rotation="vertical",
             font=font_manager.FontProperties(fname=font_path, size=10))
    
    plt.subplots_adjust(hspace=0.5, wspace=0.25)
    plt.savefig(f"{figure_path}/predicted_emotion.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_valence_correlations(speech_line_df, valence_cor_dfs, valence_var, gpt_var, figure_path,
                              conditions=["placebo", "mdma_high"], biotypes=["low", "high"]):
    
    var_i = 0
    condition2title = {"placebo": "Placebo", "mdma_high": "MDMA"}
    
    print("-"*75 + f"\n{valence_var.upper().replace("_", " ")}\n" + "-"*75)
    for condition_i, condition in enumerate(conditions):
    
        fig, axes = plt.subplots(1, 3, figsize=(2.5, 0.5), squeeze=False)
        
        df = speech_line_df[speech_line_df["condition"] == condition]
        df = df[["biotype", "block", gpt_var, valence_var]].dropna()
        max_y = df[valence_var].max()
    
        for block_i, block in enumerate([1, 5]):
            block_df = df[df["block"] == block]
            x = block_df[gpt_var]
            y = block_df[valence_var]
            axes[0,block_i].scatter(x, y, alpha=0.3, s=1, 
                                    c=[biotype2color[b] for b in block_df["biotype"]])
            m, b = np.polyfit(x, y, 1)
            axes[0,block_i].plot(x, m*x+b, linewidth=0.75, color="black") 
            stat_df = valence_cor_dfs["blocks"][condition]
            stat_df = stat_df[(stat_df["block"] == block) & (stat_df["condition"] == condition)]
            r = stat_df["r"].values[0]
            sig = stat_df["sig"].values[0]
            axes[0,block_i].set_title(f"Block {block}\n" + r"$\it{ρ}$" + f" = {r:3.2f}{sig}", 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
    
        x = df[gpt_var]
        y = df[valence_var]
        axes[0,2].scatter(x, y, alpha=0.45, s=1, c=[biotype2color[b] for b in df["biotype"]])
        m, b = np.polyfit(x, y, 1)
        axes[0,2].plot(x, m*x+b, linewidth=0.75, color="black") 
        stat_df = valence_cor_dfs["sessions"][condition]
        stat_df = stat_df[(stat_df["condition"] == condition)]
        r = stat_df["r"].values[0]
        sig = stat_df["sig"].values[0]
        axes[0,2].set_title("Session\n" + r"$\it{ρ}$" + f" = {r:3.2f}{sig}", 
                            font=font_manager.FontProperties(fname=font_path, size=9))
        
        for i in range(3):
            axes[0,i].spines["top"].set_visible(False)
            axes[0,i].spines["right"].set_visible(False)
        
            axes[0,i].set_xlim([-0.1, 1.1])
            axes[0,i].set_xticks([0, 0.5, 1])
            axes[0,i].set_xticklabels([0, 0.5, 1], 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
    
            axes[0,i].set_ylim([-0.1, 1.1])
            axes[0,i].set_yticks([0, 0.5, 1])
        
        axes[0,0].set_yticklabels([0, 0.5, 1], 
                                  font=font_manager.FontProperties(fname=font_path, size=9))
        for i in range(1,3):
            axes[0,i].set_yticklabels([], 
                                      font=font_manager.FontProperties(fname=font_path, size=9))
    
        if condition == "placebo":
            axes[0,0].set_ylabel("Regression logit", 
                                 font=font_manager.FontProperties(fname=font_path, size=9))
        
        title_y = 2.4
        axes[0,0].plot([-0.1, 4.1], [[title_y]*2, [title_y]*2], 
                       c="black", linewidth=0.75, clip_on=False)
        axes[0,0].text(2, title_y+0.15, condition2title[condition], ha="center", 
                       font=font_manager.FontProperties(fname=font_path, size=10))
        
        plt.subplots_adjust(hspace=0.965, wspace=0.25)
        plt.savefig(f"{figure_path}/{condition}_block-cors_{valence_var}_X_{gpt_var}.png", 
                    dpi=300, bbox_inches="tight")
        plt.show()
        print()




########################################################################
########################## VISUAL ANALOG SCALE #########################
########################################################################

def plot_vas_differences(speech_block_df, speech_session_df, vas_stat_dicts, vas_vars, figure_path,
                         conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                         n_blocks=5, yn=1.875, ytn=2.25, bn=0.75, cn=5.5):
    
    fig, axes = plt.subplots(len(vas_vars), 3, figsize=(6.35, len(vas_vars)), 
                             gridspec_kw={'width_ratios': [3, 3, 1.75]}, squeeze=False)
        
    for var_i, var in enumerate(vas_vars):
        
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
        
            max_y = speech_block_df[var].dropna().max()
            
            block_stat_df = vas_stat_dicts["blocks"][condition]
            block_maxes = {block: 0 for block in range(1, n_blocks+1)}
            for biotype_i, biotype in zip([-0.1, 0.1], biotypes):
                
                df = speech_block_df
                df = df[(df["condition"] == condition) & (df["biotype"] == biotype)]
                
                for block in range(1,6):
                    block_max = df[df["block"] == block][var].max()
                    if block_max > block_maxes[block]: 
                        block_maxes[block] = block_max
        
                for block in range(1, n_blocks+1):
                    c = biotype2color[biotype]
                    biotype2fill = {"high": c, "low": "none"}
                    y = df[(df["biotype"] == biotype) & (df["block"] == block)][var].dropna()
                    x = block+(biotype_i*bn)
                    p = axes[var_i,condition_i].violinplot(y, positions=[x], 
                                                           widths=[0.5], side=biotype)
                    for pc in p["bodies"]:
                        pc.set_facecolor(c)
                        pc.set_alpha(0.4)
                        pc.set_edgecolor(c)
                    for partname in ("cbars", "cmins", "cmaxes"):
                        vp = p[partname]
                        vp.set_linewidth(0)
                    axes[var_i,condition_i].plot([x, x], [y.quantile(0.25), y.quantile(0.75)], 
                                                 color=c, alpha=0.9, linewidth=1, clip_on=False)
                
                plot_blocks = block_stat_df[block_stat_df["var"] == var]["block"]
                plot_blocks = [biotype_i + b for b in plot_blocks]
                plot_meds = block_stat_df[block_stat_df["var"] == var][f"med_{biotype}"]
    
                axes[var_i,condition_i].plot(plot_blocks, plot_meds, 
                                             color=biotype2color[biotype], alpha=0.8, linewidth=1, 
                                             linestyle=biotype2linestyle[biotype]) 
    
            for block in range(1, n_blocks+1):
                for biotype, x_nudge in zip(biotypes, [-0.075, 0.075]):
                    scatter_df = speech_block_df[(speech_block_df["block"] == block) & (speech_block_df["biotype"] == biotype) & (speech_block_df["condition"] == condition)]
                    for p, val in zip(scatter_df["participant"], scatter_df[var]):
                        axes[var_i,condition_i].scatter([block + x_nudge], val, 
                                                        alpha=0.6, s=6, linewidth=0.5,
                                                        c=biotype2color[biotype], zorder=100)
            
            for block in range(1,6):
                block_dict = block_stat_df[(block_stat_df["block"] == block) & (block_stat_df["var"] == var)]
                fdr = block_dict["FDR"].values[0]
                if fdr < 0.05:
                    df = speech_block_df[(speech_block_df["condition"] == condition)][["block", var]].dropna()
                    sig = utils.p2sig(fdr).replace("***", "*\n*\n*").replace("**", "*\n*")
                    axes[var_i,condition_i].text(block, block_maxes[block]+5, sig, 
                                 clip_on=False, ha="center", linespacing=0.4,
                                 font=font_manager.FontProperties(fname=font_path, size=9))  
        
            df = speech_session_df
            df = df[df["condition"] == condition]
            for biotype_i, biotype in enumerate(biotypes):
                c = biotype2color[biotype]
                biotype2fill = {"high": c, "low": "none"}
                y = df[df["biotype"] == biotype][var].dropna()
                x = [condition_y+(biotype_i*bn)]*len(y)
                p = axes[var_i,2].violinplot(y, positions=[x[0]], widths=[3], side=biotype)
                for pc in p["bodies"]:
                    pc.set_facecolor(c)
                    pc.set_alpha(0.4)
                    pc.set_edgecolor(c)
                for partname in ("cbars", "cmins", "cmaxes"):
                    vp = p[partname]
                    vp.set_linewidth(0)
                axes[var_i,2].scatter([x[0]], [np.median(y)], 
                                      color=c, alpha=1, s=12, linewidth=1.5, 
                                      facecolors=biotype2fill[biotype])
                axes[var_i,2].plot(x[:2], [y.quantile(0.25), y.quantile(0.75)], 
                                   color=c, alpha=0.9, linewidth=1.25, clip_on=False)
        
        n_sig = 0
        for condition_i, condition_y, condition in zip([0, 1], [0, cn], ["placebo", "mdma_high"]):
            star_y = max_y+yn+(n_sig*yn*ytn)+3
            condition_stat_df = vas_stat_dicts["sessions"][condition]
            fdr = condition_stat_df["FDR"].values[0]
            if fdr < 0.05:
                axes[var_i,2].plot([condition_y, condition_y+bn], [star_y]*2, 
                                   c="black", linewidth=0.75, clip_on=False)
                axes[var_i,2].text(condition_y+(bn*0.5), star_y-0.5, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                   font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
        for biotype_i, biotype in enumerate(biotypes):
            y = df[df["biotype"] == biotype][var].dropna()
            x = [(biotype_i*bn)]*len(y)
            biotype_stat_df = vas_stat_dicts["sessions"][biotype]
            fdr = biotype_stat_df["FDR"].values[0]
            if fdr < 0.05:
                mid_x = x[0] + (x[0]+cn - x[0])*0.5
                star_y = max_y+(yn)+(n_sig*yn*ytn)+12
                axes[var_i,2].plot([x[0], x[0]+cn], [star_y]*2, c="black", linewidth=0.75, clip_on=False)
                axes[var_i,2].text(mid_x, star_y-0.5, utils.p2sig(fdr), 
                                   clip_on=False, ha="center",
                                   font=font_manager.FontProperties(fname=font_path, size=9))
                n_sig += 1
    
    
        title_y = max_y+yn+(n_sig*yn*ytn)+21.5
        
        axes[var_i,0].plot([1, 5], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
        axes[var_i,0].text(3, title_y+5, "Placebo", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
            
        axes[var_i,1].plot([1, 5], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
        axes[var_i,1].text(3, title_y+5, "MDMA", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
        
        axes[var_i,2].plot([-2, 2.8], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
        axes[var_i,2].plot([3.5, 8.3], [[title_y]*2, [title_y]*2], 
                           c="black", linewidth=0.75, clip_on=False)
        axes[var_i,2].text(0.4, title_y+5, "Placebo", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
        axes[var_i,2].text(5.9, title_y+5, "MDMA", ha="center", 
                           font=font_manager.FontProperties(fname=font_path, size=10))
        
        for i in range(2):
            axes[var_i,i].set_xlim([0.5, 5.5])
            axes[var_i,i].set_xticks([1, 2, 3, 4, 5])
            axes[var_i,i].set_xticklabels([1, 2, 3, 4, 5], 
                                          font=font_manager.FontProperties(fname=font_path, size=9))
            axes[var_i,i].set_xlabel("Block", 
                                     font=font_manager.FontProperties(fname=font_path, size=10))
        
        for i in range(3):
            axes[var_i,i].set_ylim([40, 110])
            axes[var_i,i].set_yticks([50, 75, 100])
            axes[var_i,i].set_yticklabels([50, 75, 100], 
                                          font=font_manager.FontProperties(fname=font_path, size=9))
            axes[var_i,i].spines["top"].set_visible(False)
            axes[var_i,i].spines["right"].set_visible(False)
    
        axes[var_i,2].spines["bottom"].set_visible(False)
        axes[var_i,2].set_xlim([-3, 7.75])
        axes[var_i,2].set_xticks([])
        axes[var_i,2].set_xticklabels([])
    
        if var_i == len(vas_vars)-1:
            label_y = 45
            axes[var_i,2].text(-3.5, label_y, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(-1.5, label_y, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(2, label_y, r"NTN$_{\rm A–}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
            axes[var_i,2].text(4, label_y, r"NTN$_{\rm A+}$", 
                               ha="left", va="top", rotation=60, 
                               font=font_manager.FontProperties(fname=font_path, size=10))
    
    axes[var_i,0].set_ylabel("VAS rating", 
                             font=font_manager.FontProperties(fname=font_path, size=10))
    
    plt.subplots_adjust(hspace=0.965, wspace=0.25)
    plt.savefig(f"{figure_path}/{var}.png", dpi=300, bbox_inches="tight")
    plt.show()
    print()


def plot_vas_correlations(speech_session_df, vas_cor_df, vas_vars, gpt_var, figure_path,
                          conditions=["placebo", "mdma_high"], biotypes=["low", "high"],
                          yn=1.875, ytn=2.25):
    
    for var_i, var in enumerate(vas_vars):
        print("-"*75 + f"\n{var.upper().replace("_", " ")}\n" + "-"*75)
        fig, axes = plt.subplots(1, 2, figsize=(1.75, 0.5), squeeze=False)
        for condition_i, condition in enumerate(conditions):
    
            max_y = 100
    
            df = speech_session_df
            df = df[df["condition"] == condition]
    
            for x, y, b, p in zip(df[gpt_var], df[var], df["biotype"], df["participant"]):
                axes[0,condition_i].scatter(x, y, alpha=0.75, s=10, c=biotype2color[b])
    
            m, b = np.polyfit(df[gpt_var], df[var], 1)
            axes[0,condition_i].plot(df[gpt_var], m*df[gpt_var]+b, 
                                     linewidth=0.75, color="black") 
            
            axes[0,condition_i].spines["top"].set_visible(False)
            axes[0,condition_i].spines["right"].set_visible(False)
    
            axes[0,condition_i].set_xlim([-0.1, 0.6])
            axes[0,condition_i].set_xticks([0, 0.3, 0.6])
            axes[0,condition_i].set_xticklabels([0, 0.3, 0.6], 
                                                font=font_manager.FontProperties(fname=font_path, size=9))
    
            stat_df = vas_cor_df[condition]
            stat_df = stat_df[stat_df["var_x"] == gpt_var]
            stat_df = stat_df[stat_df["var_y"] == var]
            r = stat_df["r"].values[0]
            sig = stat_df["sig"].values[0]
            axes[0,condition_i].set_title(r"$\it{ρ}$" + f" = {r:3.2f}{sig}", 
                                          font=font_manager.FontProperties(fname=font_path, size=9))
    
        for i in range(2):
            axes[0,i].set_ylim([-10, 110])
            axes[0,i].set_yticks([0, 50, 100])
        axes[0,0].set_yticklabels([0, 50, 100], 
                                  font=font_manager.FontProperties(fname=font_path, size=9))
        axes[0,1].set_yticklabels([], 
                                  font=font_manager.FontProperties(fname=font_path, size=9))
    
        title_y = max_y+yn+(yn*ytn)+80
        axes[0,0].plot([-0.1, 0.6], [[title_y]*2, [title_y]*2], 
                       c="black", linewidth=0.75, clip_on=False)
        axes[0,0].text(0.25, title_y+10, "Placebo", ha="center", 
                       font=font_manager.FontProperties(fname=font_path, size=10))
            
        axes[0,1].plot([-0.1, 0.6], [[title_y]*2, [title_y]*2], 
                       c="black", linewidth=0.75, clip_on=False)
        axes[0,1].text(0.25, title_y+10, "MDMA", ha="center", 
                       font=font_manager.FontProperties(fname=font_path, size=10))
        
        plt.subplots_adjust(hspace=0.965, wspace=0.25)
        plt.savefig(f"{figure_path}/{var}_cors_{gpt_var}.png", 
                    dpi=300, bbox_inches="tight")
        plt.show()
        print()
        