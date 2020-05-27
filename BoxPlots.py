import matplotlib.pyplot as plt

complete_utility_ndcg5 = [0.7734,0.8034,0.8332,0.7975,0.7824]
complete_utility_ndcg10 = [0.803,0.8272,0.8444,0.8309,0.8233]
complete_rel_ndcg5 = [0.5799,0.5855,0.6513,0.559,0.5753]
complete_rel_ndcg10 = [0.6466,0.6247,0.6911,0.6202,0.6303]
complete_imp_ndcg5 = [0.8438,0.8381,0.8122,0.8575,0.8095]
complete_imp_ndcg10 = [0.844,0.8306,0.8057,0.8607,0.8313]

y = [complete_utility_ndcg5,complete_utility_ndcg10,complete_rel_ndcg5,complete_rel_ndcg10,complete_imp_ndcg5,complete_imp_ndcg10]
x = ["utility ndcg@5","utility ndcg@10","rel ndcg@5","rel ndcg@10","imp ndcg@5","imp ndcg@10"]
plt.boxplot(y, labels=x)


plt.show()