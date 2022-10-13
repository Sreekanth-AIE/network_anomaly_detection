# tools for handling n dimentional data or martix type of data
import numpy as np
import pandas as pd

# tools for EDA and data visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
# plt.rcParams['figure.figsize'] = (4,3)
# plt.rcParams['font.family'] = 'sans-serif'

# for model evaluation
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve

def univ_confusion_matrix(actual_values,predicted_values,classes,fmt="d",
                          cmap="Blues",annot=True,outline_color="black",line_color=None):
    cfm = confusion_matrix(actual_values,predicted_values) 
    ax = sns.heatmap(data=cfm,annot=annot,fmt=fmt,cmap=cmap,yticklabels=classes,xticklabels=classes,
                     linewidths=0.4,linecolor=line_color)
    ax.axvline(x=0,color=outline_color)
    ax.axvline(x=cfm.shape[0],color=outline_color)
    ax.axhline(y=0,color=outline_color)
    ax.axhline(y=cfm.shape[1],color=outline_color)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

def gen_test_report(result_df):
    eval_metrics_data = result_df.drop(["Training_time","Testing_time"],axis=1)
    eval_metrics_data.set_index("Algorithm",drop=True,inplace=True)
    ax1 = eval_metrics_data.plot(kind="bar",figsize=(25,4),width=0.8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=5)
    for i in ax1.containers:
        plt.bar_label(i,fmt='%.2f', label_type='center')
    plt.title('Performance of Models by Classification Metric')
    plt.xticks(rotation=0)
    # plt.savefig('models_performance.png',dpi=300)

    time_metrics_data = result_df[["Training_time","Testing_time","Algorithm"]]
    time_metrics_data.set_index("Algorithm",drop=True,inplace=True)
    ax2 = time_metrics_data.plot(kind="bar",figsize=(19,4),width=0.8)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),fancybox=True, shadow=True, ncol=5)
    for i in ax2.containers:
        plt.bar_label(i,fmt='%.4f', label_type='edge')
    plt.title('Time Consumption of Models')
    plt.xticks(rotation=0)
    # plt.savefig('time_performance.png',dpi=300)

    plt.tight_layout()

def univ_ROC_curve(y_test,  y_pred):
    fpr, tpr, _ = roc_curve(y_test,  y_pred)
    auc = roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,label="classify, auc="+str(auc))
    plt.legend(loc=4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True POsitive Rate")
    plt.title("ROC Curve of Selected Model")

def model_performance_graph_rep(y_test,  y_pred, classes,model_name="selected"):
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    univ_confusion_matrix(y_pred,y_test,classes)
    plt.title(f"Confusion Matrix of {model_name} Model")

    plt.subplot(1,2,2)
    univ_ROC_curve(y_test,  y_pred)
    plt.title(f"ROC Curve of {model_name} Model")

    plt.suptitle(f"Graphical Report on {model_name} Model")
    plt.tight_layout()
    plt.show()
    # plt.savefig('model_performance.png',dpi=300)


# For future dev

# import matplotlib.pyplot as plt
# from matplotlib import gridspec 
# import seaborn as sns
# # plt.figure(figsize=(20, 12))
# # sns.set(font_scale=2.5)

# fig = plt.figure(figsize=(22, 50))

# # spec = gridspec.GridSpec(ncols=2, nrows=8)

# plt.subplot(8,2,1)
# eval_metrics_data=eval_results_df.drop(["fit_time","score_time"],axis=1)
# bootst_eval_df = pd.melt(eval_metrics_data,id_vars=["Algorithm"],var_name="metrics",value_name="values")

# sns.boxplot(x="Algorithm", y="values", hue="metrics", data=bootst_eval_df, palette="Set3")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.title('Cross-Validation of Models by Classification Metric')
# plt.xticks(rotation=90)
# # plt.show()
# # plt.savefig('./benchmark_models_performance.png',dpi=300)

# plt.subplot(8,2,3).sharey(plt.subplot(8,2,3))
# time_metrics_data=eval_results_df[["fit_time","score_time","Algorithm"]]
# bootst_time_df = pd.melt(time_metrics_data,id_vars=["Algorithm"],var_name="metrics",value_name="values")

# sns.boxplot(x="Algorithm", y="values", hue="metrics", data=bootst_time_df, palette="Set3")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.title('Time Taken for Cross-Validation of Models in Secs')
# plt.xticks(rotation=90)
# plt.tight_layout()

# plt.subplot(8,2,5)
# ax = sns.barplot(data=test_results_df,x="Values",y="metrics")
# ax.bar_label(ax.containers[0],fmt='%.2f', label_type='center')
# plt.title(f"""performance metrics of Best Model "{best_model_name}" """)

# plt.subplot(8,2,7)
# utils.univ_confusion_matrix(y_pred,y_test,["beng","mal"])
# plt.title("Confusion Matrix of Best Model")

# plt.subplot(8,2,8)
# fpr, tpr, _ = roc_curve(y_test,  y_pred)
# auc = roc_auc_score(y_test, y_pred)
# plt.plot(fpr,tpr,label="classify, auc="+str(auc))
# plt.legend(loc=4)
# plt.xlabel("False Positive Rate")
# plt.ylabel("True POsitive Rate")
# plt.title("ROC Curve of Best Model")
# plt.tight_layout()

# plt.tight_layout()
# plt.show()
# # plt.savefig('./benchmark_models_performance.png',dpi=300)