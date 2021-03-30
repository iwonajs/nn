import pickle

pos_path = "/home/iwona/results_out/car_models/CAR_variant_32_epochs100_r2/metrics_val_loss_min/fold_0/results/model/val_loss/predictions_positive_-_metrics_val_loss_min_-_val_loss.p"
neg_path = "/home/iwona/results_out/car_models/CAR_variant_32_epochs100_r2/metrics_val_loss_min/fold_0/results/model/val_loss/predictions_negative_-_metrics_val_loss_min_-_val_loss.p"

pos = pickle.load(open(pos_path, "rb"))
neg = pickle.load(open(neg_path, "rb"))

print("Len Pos: ", len(pos))
print("Len Neg: ", len(neg))

TP = sum([1 if i >= 0.50 else 0 for i in pos])
FN = sum([1 if i < 0.50 else 0 for i in pos])
false_negatives_boundary = [i for i in pos if i < 0.50 and i > 0.40]
TPR = TP/len(pos)
print("POS:", TP, FN, len(false_negatives_boundary))
print("POS %:", TPR, FN/len(pos), len(false_negatives_boundary)/len(pos))


TN = sum([1 if i < 0.50 else 0 for i in neg])
FP = sum([1 if i >= 0.50 else 0 for i in neg])
FP_list = [i if i >= 0.50 else 0 for i in neg]
TNR = TN/len(neg)
#false_negatives_boundary = [i for i in pos if i < 0.50 and i > 0.40]
print("NEG:", TN, FP)
print("NEG:", TNR, FP/len(neg))
#print(false_negatives_boundary)

PPV = TP / (TP + FP)
#FP = 0
print("TP rate (sensitivity): ", TPR)
print("TN rate (specificity): ", TNR)
print("PPV, (precision): ", PPV)
print("Fscore:", 2*TP / (2*TP + FP + FN))
print("TP, FP:", TP, FP)
print("FN, TN: ", FN, TN)
print("TP, FP:", round(TP/len(pos),2), round(FP/len(neg),2))
print("FN, TN: ", round(FN/len(pos),2), round(TN/len(neg),2))
#print(FP_list)

# Len Pos:  761
# Len Neg:  80666
# POS: 464 297 137
# POS %: 0.6097240473061761 0.3902759526938239 0.1800262812089356
# NEG: 52650 28016
# NEG: 0.6526913445565666 0.3473086554434334
# TP rate (sensitivity):  0.6097240473061761
# TN rate (specificity):  0.6526913445565666
# PPV, (precision):  0.016292134831460674
# Fscore: 0.031736260729797204
# TP, FP: 464 28016
# FN, TN:  297 52650



