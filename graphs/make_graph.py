import pandas as pd
from matplotlib import pyplot as plt
import os


epochs = [1, 2, 3, 4]
accuracies = [
    [0.6153, 0.7827, 0.9051, 0.9415],
    [0.6034, 0.7870, 0.8143, 0.8651],
    [0.4993, 0.5018, 0.5102, 0.4909],
    [0.5761, 0.6377, 0.6741, 0.7159],
    [0.5996, 0.7255, 0.7632, 0.7266],
    [0.5793, 0.6940, 0.6655, 0.6662],
    [0.5138, 0.5248, 0.5309, 0.5577]
]

losses = [
    [23.7541, 1.5175, 0.5799, 0.3474],
    [10.6693, 1.0355, 1.0308, 0.5771],
    [2.1408, 0.8072, 0.8598, 1.6904],
    [9.3036, 3.3187, 2.3174, 2.3518],
    [2.259, 0.7, 0.6195, 1.1463],
    [1.4019, 0.6737, 1.21, 1.3821],
    [1.1817, 0.7732, 0.7981, 0.7138]
]

labels = [
    "trunct_1_bs_25_epochs_4_data5%_learning0001",
    "trunct_2_bs_25_epochs_4_data5%_learning0001",
    "trunct_7_bs_25_epochs_4_data5%_learning0001",
    "trunct_3_bs_25_epochs_4_data5%_learning0001",
    "trunct_4_bs_25_epochs_4_data5%_learning0001",
    "trunct_5_bs_25_epochs_4_data5%_learning0001",
    "trunct_9_bs_25_epochs_4_data5%_learning0001"
]

max_accuracy = {}
min_loss = {}

for i in range(len(labels)):
    accuracy, loss, label = accuracies[i], losses[i], labels[i]
    trunc_at = int(label.split("_")[1])
    max_accuracy[trunc_at] = max(accuracy)
    min_loss[trunc_at] = min(loss)
    if os.path.exists(label + "_loss.png") and os.path.exists(label + "_accuracy.png"):
        continue
    accuracy_df, loss_df = pd.DataFrame(), pd.DataFrame()
    accuracy_df["epochs"], loss_df["epochs"] = epochs, epochs
    accuracy_df["accuracies"] = accuracy
    loss_df["losses"] = loss
    # plotting accuracy and save
    accuracy_df.plot(x='epochs', y='accuracies')
    plt.title(label + " accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.savefig(label + "_accuracy.png")
    # plotting loss and save
    loss_df.plot(x='epochs', y='losses')
    plt.title(label + " loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig(label + "_loss.png")

# alright, now plot barplot:
truncs = list(sorted(max_accuracy.keys()))
accuracies = [max_accuracy[e] for e in truncs]
losses = [min_loss[e] for e in truncs]

agg_accuracies, agg_losses = pd.DataFrame(), pd.DataFrame()
agg_accuracies["truncates"], agg_losses["truncates"] = truncs, truncs
agg_accuracies["accuracies"], agg_losses["losses"] = accuracies, losses

agg_accuracies.plot.bar(x="truncates", y="accuracies")
plt.xlabel("truncates")
plt.ylabel("accuracy")
plt.title("max accuracy per truncate val")
plt.savefig("cross_model_accuracies.png")





agg_losses.plot.bar(x="truncates", y="losses")
plt.xlabel("truncates")
plt.ylabel("loss")
plt.title("min loss per truncate val")
plt.savefig("cross_model_losses.png")